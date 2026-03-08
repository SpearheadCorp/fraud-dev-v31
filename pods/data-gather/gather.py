"""
Pod 1: Data Gather
Generates synthetic credit card transaction data seeded from real statistical distributions.
Supports 'once' and 'continuous' run modes with hot-reload stress config.
"""
import os
import sys
import time
import signal
import logging
import zipfile
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker
from scipy import stats
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    log.info("[INFO] Signal %s received — shutting down gracefully", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/raw"))  # legacy fallback
OUTPUT_PATH_GPU = Path(os.environ.get("OUTPUT_PATH_GPU", "/data/raw/gpu"))
OUTPUT_PATH_CPU = Path(os.environ.get("OUTPUT_PATH_CPU", "/data/raw/cpu"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", str(max(1, (os.cpu_count() or 2) // 2))))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
FRAUD_RATE = float(os.environ.get("FRAUD_RATE", "0.005"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "1000000"))
RUN_MODE = os.environ.get("RUN_MODE", "once")
STRESS_MODE = os.environ.get("STRESS_MODE", "false").lower() == "true"
KAGGLE_SEED_PATH = os.environ.get("KAGGLE_SEED_PATH", "")
STRESS_CONFIG_PATH = Path(os.environ.get("STRESS_CONFIG_PATH", "/data/stress/stress.conf"))
# Rate governor: target rows/sec with asymmetric jitter for realistic variation.
# 0 = unlimited (run as fast as possible). Overridden per-job via env.
TARGET_ROWS_PER_SEC = int(os.environ.get("TARGET_ROWS_PER_SEC", "0"))

# Identity pool sizes — shared across all workers so the same users/merchants
# appear repeatedly across chunks, giving the GNN graph meaningful connectivity.
NUM_USERS = 10_000     # unique cardholders (~100 txns/user over 1M rows)
NUM_MERCHANTS = 1_000  # unique merchants

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
    "health_fitness", "travel", "kids_pets", "home",
]
HIGH_FRAUD_CATEGORIES = {"shopping_net", "travel", "misc_net"}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

# Hardcoded fallback distributions (used if seed file unavailable)
_HARDCODED_DEFAULTS: dict = {
    "fraud_rate": 0.005,
    "amount_legit_params": (0.8, 0.0, 45.0),    # lognorm (s, loc, scale)
    "amount_fraud_params": (1.0, 0.0, 150.0),
    "hour_weights": [
        0.030, 0.020, 0.020, 0.020, 0.020, 0.030,   # 00-05
        0.040, 0.050, 0.060, 0.060, 0.060, 0.060,   # 06-11
        0.060, 0.060, 0.060, 0.060, 0.060, 0.060,   # 12-17
        0.060, 0.050, 0.050, 0.040, 0.040, 0.030,   # 18-23
    ],
    "fraud_hour_weights": [
        0.120, 0.100, 0.080, 0.070, 0.070, 0.080,
        0.060, 0.050, 0.050, 0.040, 0.040, 0.040,
        0.040, 0.040, 0.040, 0.040, 0.040, 0.040,
        0.030, 0.030, 0.030, 0.040, 0.040, 0.040,
    ],
    "category_weights": [1 / 14] * 14,
    "fraud_category_weights": [1 / 14] * 14,
    "lat_range": (24.0, 50.0),
    "long_range": (-125.0, -65.0),
    "merch_lat_range": (23.0, 51.0),
    "merch_long_range": (-126.0, -64.0),
    "city_pop_params": (1.2, 0.0, 8000.0),
    "unix_range": (1325376000, 1388534400),
    "zip_range": (1001, 99950),
}


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

def _build_identity_pools() -> tuple:
    """
    Build fixed cardholder (cc_num) and merchant name pools shared across all
    workers via the dist dict.  Ensures repeat users/merchants across chunks so
    the GNN graph has meaningful connectivity (same card → multiple transaction
    nodes, same merchant → multiple transaction nodes).
    """
    rng = np.random.default_rng(42)
    faker = Faker("en_US")
    Faker.seed(42)
    cc_num_pool = rng.integers(10**15, 10**16 - 1, NUM_USERS)
    merchant_pool = [
        "fraud_" + faker.company().replace(",", "").replace(" ", "_")[:28]
        for _ in range(NUM_MERCHANTS)
    ]
    return cc_num_pool, merchant_pool


def _open_csv(seed_path: Path) -> pd.DataFrame:
    """Open CSV directly or from inside a zip archive."""
    if seed_path.suffix == ".zip":
        with zipfile.ZipFile(seed_path) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {seed_path}")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(seed_path)


def load_seed_distributions(seed_csv_path: str) -> dict:
    """
    Fit statistical distributions from seed transaction data.
    Returns distribution params used by generator — NOT the raw data.

    Supports both:
      - Sparkov schema (amt, unix_time, lat, long, is_fraud, category, ...)
      - Classic Kaggle CC schema (Time, V1-V28, Amount, Class)
    """
    path = Path(seed_csv_path)
    log.info("[INFO] Loading seed distributions from: %s", path)
    try:
        df = _open_csv(path)

        # Detect schema and normalise column names
        if "Class" in df.columns and "Amount" in df.columns:
            # Classic Kaggle CC Fraud schema
            df = df.rename(columns={"Class": "is_fraud", "Amount": "amt"})
            # Simulate unix_time from Time column (seconds elapsed, start ~2013-09-01)
            df["unix_time"] = df["Time"].astype(int) + 1378080000
            df["lat"] = np.nan
            df["long"] = np.nan
            df["merch_lat"] = np.nan
            df["merch_long"] = np.nan
            df["zip"] = 10000
            df["city_pop"] = 5000
            df["category"] = "misc_net"
        elif "is_fraud" not in df.columns:
            raise ValueError("Cannot detect schema: missing 'is_fraud' or 'Class' column")

        fraud = df[df["is_fraud"] == 1]
        legit = df[df["is_fraud"] == 0]
        fraud_rate = len(fraud) / len(df)
        log.info("[INFO] Seed: %d rows, fraud_rate=%.4f", len(df), fraud_rate)

        amount_legit = stats.lognorm.fit(legit["amt"].clip(lower=0.01))
        amount_fraud = stats.lognorm.fit(fraud["amt"].clip(lower=0.01))

        hours_all = (df["unix_time"] % 86400) // 3600
        hours_fraud = (fraud["unix_time"] % 86400) // 3600
        hour_hist, _ = np.histogram(hours_all, bins=24, range=(0, 24), density=True)
        fraud_hour_hist, _ = np.histogram(hours_fraud, bins=24, range=(0, 24), density=True)
        hour_weights = (hour_hist / hour_hist.sum()).tolist()
        fraud_hour_weights = (fraud_hour_hist / fraud_hour_hist.sum()).tolist()

        cat_counts = df["category"].value_counts(normalize=True)
        fraud_cat_counts = fraud["category"].value_counts(normalize=True)
        category_weights = cat_counts.reindex(ALL_CATEGORIES).fillna(0.001).tolist()
        fraud_category_weights = fraud_cat_counts.reindex(ALL_CATEGORIES).fillna(0.0001).tolist()

        # Geo ranges — fall back to hardcoded if NaN
        def safe_range(series, default):
            clean = series.dropna()
            return (float(clean.min()), float(clean.max())) if len(clean) > 0 else default

        lat_range = safe_range(df["lat"], _HARDCODED_DEFAULTS["lat_range"])
        long_range = safe_range(df["long"], _HARDCODED_DEFAULTS["long_range"])
        merch_lat_range = safe_range(df["merch_lat"], _HARDCODED_DEFAULTS["merch_lat_range"])
        merch_long_range = safe_range(df["merch_long"], _HARDCODED_DEFAULTS["merch_long_range"])

        city_pop_params = stats.lognorm.fit(df["city_pop"].clip(lower=1))
        unix_range = (int(df["unix_time"].min()), int(df["unix_time"].max()))
        zip_range = (int(df["zip"].min()), int(df["zip"].max()))

        return {
            "fraud_rate": fraud_rate,
            "amount_legit_params": amount_legit,
            "amount_fraud_params": amount_fraud,
            "hour_weights": hour_weights,
            "fraud_hour_weights": fraud_hour_weights,
            "category_weights": category_weights,
            "fraud_category_weights": fraud_category_weights,
            "lat_range": lat_range,
            "long_range": long_range,
            "merch_lat_range": merch_lat_range,
            "merch_long_range": merch_long_range,
            "city_pop_params": city_pop_params,
            "unix_range": unix_range,
            "zip_range": zip_range,
        }
    except Exception as exc:
        log.warning("[WARN] Could not load seed data (%s) — using hardcoded defaults", exc)
        return _HARDCODED_DEFAULTS.copy()


# ---------------------------------------------------------------------------
# Per-worker state (set by worker_init, used by generate_chunk)
# ---------------------------------------------------------------------------
_DIST: dict = {}
_FAKER: Optional[Faker] = None


def _worker_init(dist: dict) -> None:
    """Initialise per-worker globals once when the worker process starts."""
    global _DIST, _FAKER
    _DIST = dist
    _FAKER = Faker("en_US")
    Faker.seed(os.getpid())
    np.random.seed(os.getpid())


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_chunk(args: tuple) -> tuple:
    """
    Generate one chunk of synthetic transactions.
    Args: (chunk_id, n_rows, seed_offset)
    Returns: (DataFrame, chunk_id)
    """
    chunk_id, n_rows, seed_offset = args
    dist = _DIST
    faker = _FAKER
    rng = np.random.default_rng(seed_offset + chunk_id * 997)

    fraud_rate = dist.get("fraud_rate", 0.005)
    hour_weights = np.array(dist.get("hour_weights", _HARDCODED_DEFAULTS["hour_weights"]))
    fraud_hour_weights = np.array(
        dist.get("fraud_hour_weights", _HARDCODED_DEFAULTS["fraud_hour_weights"])
    )
    cat_w = np.array(dist.get("category_weights", [1 / 14] * 14))
    fraud_cat_w = np.array(dist.get("fraud_category_weights", [1 / 14] * 14))

    # Normalize probability arrays to be safe
    hour_weights = hour_weights / hour_weights.sum()
    fraud_hour_weights = fraud_hour_weights / fraud_hour_weights.sum()
    cat_w = cat_w / cat_w.sum()
    fraud_cat_w = fraud_cat_w / fraud_cat_w.sum()

    # --- Fraud flags (base) ---
    is_fraud = rng.random(n_rows) < fraud_rate

    # --- Category (fraud gets fraud-category distribution) ---
    categories = np.where(
        is_fraud,
        rng.choice(ALL_CATEGORIES, size=n_rows, p=fraud_cat_w),
        rng.choice(ALL_CATEGORIES, size=n_rows, p=cat_w),
    )

    # Boost fraud rate for high-fraud categories (3x)
    in_high_fraud_cat = np.isin(categories, list(HIGH_FRAUD_CATEGORIES))
    extra_fraud = (~is_fraud) & in_high_fraud_cat & (rng.random(n_rows) < fraud_rate * 2)
    is_fraud = is_fraud | extra_fraud

    # --- Hours (fraud skews to night 00-05) ---
    hours = np.where(
        is_fraud,
        rng.choice(24, size=n_rows, p=fraud_hour_weights),
        rng.choice(24, size=n_rows, p=hour_weights),
    )

    # --- Unix timestamps ---
    unix_range = dist.get("unix_range", _HARDCODED_DEFAULTS["unix_range"])
    base_unix = rng.integers(unix_range[0], unix_range[1], n_rows)
    unix_time = (base_unix - (base_unix % 86400)) + hours * 3600 + rng.integers(0, 3600, n_rows)

    # --- Amounts ---
    legit_p = dist.get("amount_legit_params", _HARDCODED_DEFAULTS["amount_legit_params"])
    fraud_p = dist.get("amount_fraud_params", _HARDCODED_DEFAULTS["amount_fraud_params"])
    rng_state_legit = int(rng.integers(0, 2**30))
    rng_state_fraud = int(rng.integers(0, 2**30))
    amt_legit = stats.lognorm.rvs(*legit_p, size=n_rows, random_state=rng_state_legit)
    amt_fraud = stats.lognorm.rvs(*fraud_p, size=n_rows, random_state=rng_state_fraud)
    amt = np.where(is_fraud, amt_fraud, amt_legit)
    amt = np.clip(amt, 1.0, 30000.0).round(2)

    # --- Geographic coordinates ---
    lat_r = dist.get("lat_range", _HARDCODED_DEFAULTS["lat_range"])
    lon_r = dist.get("long_range", _HARDCODED_DEFAULTS["long_range"])
    m_lat_r = dist.get("merch_lat_range", _HARDCODED_DEFAULTS["merch_lat_range"])
    m_lon_r = dist.get("merch_long_range", _HARDCODED_DEFAULTS["merch_long_range"])

    lat = rng.uniform(lat_r[0], lat_r[1], n_rows)
    long_ = rng.uniform(lon_r[0], lon_r[1], n_rows)

    # Normal merchants near customer; fraud merchants far away (> ~500 km offset)
    merch_lat_normal = rng.uniform(m_lat_r[0], m_lat_r[1], n_rows)
    merch_lon_normal = rng.uniform(m_lon_r[0], m_lon_r[1], n_rows)
    far_lat = lat + rng.uniform(4.5, 12.0, n_rows) * rng.choice([-1, 1], n_rows)
    far_lon = long_ + rng.uniform(5.0, 18.0, n_rows) * rng.choice([-1, 1], n_rows)

    merch_lat = np.where(is_fraud, np.clip(far_lat, 23.0, 51.0), merch_lat_normal)
    merch_long = np.where(is_fraud, np.clip(far_lon, -126.0, -64.0), merch_lon_normal)

    # --- City population ---
    cpp = dist.get("city_pop_params", _HARDCODED_DEFAULTS["city_pop_params"])
    city_pop = np.clip(
        stats.lognorm.rvs(*cpp, size=n_rows, random_state=int(rng.integers(0, 2**30))),
        100, 5_000_000,
    ).astype(int)

    # --- Zip codes ---
    zip_r = dist.get("zip_range", _HARDCODED_DEFAULTS["zip_range"])
    zip_codes = rng.integers(zip_r[0], zip_r[1], n_rows)

    # Online merchants (high-fraud categories) often have no merch_zipcode
    merch_zipcode = np.where(in_high_fraud_cat, np.nan, zip_codes.astype(float))

    # --- States ---
    states = np.array(US_STATES)[rng.integers(0, len(US_STATES), n_rows)]

    # --- Genders ---
    genders = rng.choice(["M", "F"], n_rows, p=[0.45, 0.55])

    # --- Transaction IDs ---
    trans_nums = [f"{rng.integers(10**15, 10**16 - 1):016x}" for _ in range(n_rows)]

    # --- Dates ---
    trans_dates = pd.to_datetime(unix_time, unit="s").strftime("%Y-%m-%d %H:%M:%S")

    # --- Names, addresses (pool-based for performance) ---
    pool_size = min(500, n_rows)
    first_pool = [faker.first_name() for _ in range(pool_size)]
    last_pool = [faker.last_name() for _ in range(pool_size)]
    street_pool = [faker.street_address()[:50] for _ in range(pool_size)]
    city_pool_names = [faker.city()[:30] for _ in range(pool_size)]
    job_pool = [faker.job()[:40] for _ in range(pool_size)]

    # Use shared merchant pool from dist for cross-chunk graph connectivity.
    # Fall back to per-chunk generation if pool not available.
    shared_merchant_pool = dist.get("merchant_pool")
    if shared_merchant_pool:
        merchant_pool = shared_merchant_pool
    else:
        merchant_pool = [
            "fraud_" + faker.company().replace(",", "").replace(" ", "_")[:28]
            for _ in range(pool_size)
        ]

    fi = rng.integers(0, pool_size, n_rows)
    li = rng.integers(0, pool_size, n_rows)
    mi = rng.integers(0, len(merchant_pool), n_rows)
    si = rng.integers(0, pool_size, n_rows)
    ci = rng.integers(0, pool_size, n_rows)
    ji = rng.integers(0, pool_size, n_rows)

    # DOB: 18-80 years before first unix_time in range
    dob_min = unix_range[0] - 80 * 365 * 86400
    dob_max = unix_range[0] - 18 * 365 * 86400
    dob_unix = rng.integers(max(dob_min, 0), dob_max, n_rows)
    dob = pd.to_datetime(dob_unix, unit="s").strftime("%Y-%m-%d")

    # Sample from shared cc_num pool so the same cardholders reappear across
    # chunks — required for GNN graph connectivity.
    cc_num_pool = dist.get("cc_num_pool")
    if cc_num_pool is not None and len(cc_num_pool) > 0:
        cc_nums = np.asarray(cc_num_pool)[rng.integers(0, len(cc_num_pool), n_rows)]
    else:
        cc_nums = rng.integers(10**15, 10**16 - 1, n_rows)

    df = pd.DataFrame(
        {
            "trans_date_trans_time": trans_dates,
            "cc_num": cc_nums,
            "merchant": [merchant_pool[i] for i in mi],
            "category": categories,
            "amt": amt,
            "first": [first_pool[i] for i in fi],
            "last": [last_pool[i] for i in li],
            "gender": genders,
            "street": [street_pool[i] for i in si],
            "city": [city_pool_names[i] for i in ci],
            "state": states,
            "zip": zip_codes,
            "lat": lat,
            "long": long_,
            "city_pop": city_pop,
            "job": [job_pool[i] for i in ji],
            "dob": dob,
            "trans_num": trans_nums,
            "unix_time": unix_time,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "is_fraud": is_fraud.astype(np.int8),
            "merch_zipcode": merch_zipcode,
        }
    )
    return df, chunk_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_disk_space(path: Path) -> tuple:
    """Returns (usage_fraction, should_stop).

    FlashBlade thin-provisioning reports df as 100% even when writes succeed.
    Use a test write to detect true ENOSPC rather than relying on df percent.
    """
    try:
        check_path = path if path.exists() else path.parent
        usage = psutil.disk_usage(str(check_path))
        pct = usage.percent / 100.0
        # Test-write to detect real ENOSPC (df can report 100% on FlashBlade
        # thin-provisioned volumes even when writes still succeed).
        test_file = check_path / ".disk_check_probe"
        try:
            test_file.write_bytes(b"x")
            test_file.unlink(missing_ok=True)
            return pct, False  # write succeeded — not truly full
        except OSError:
            return pct, True   # real ENOSPC
    except Exception:
        return 0.0, False


def load_stress_config() -> dict:
    """Hot-reload stress config from shared file."""
    if not STRESS_CONFIG_PATH.exists():
        return {}
    try:
        config: dict = {}
        for line in STRESS_CONFIG_PATH.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
        return config
    except Exception:
        return {}


def emit_telemetry(total_rows: int, total_bytes: float, files_written: int,
                   num_workers: int, fraud_rate: float, start_time: float,
                   rows_since_last: int = 0, elapsed_since_last: float = 1.0) -> None:
    elapsed = max(time.time() - start_time, 0.001)
    throughput_mbps = (total_bytes / 1e6) / elapsed
    rows_per_sec = int(rows_since_last / max(elapsed_since_last, 0.001))
    sys.stdout.write(
        f"[TELEMETRY] stage=gather rows_generated={total_rows} "
        f"rows_per_sec={rows_per_sec} "
        f"throughput_mbps={throughput_mbps:.1f} files_written={files_written} "
        f"workers={num_workers} fraud_rate={fraud_rate:.4f}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _SHUTDOWN

    OUTPUT_PATH_GPU.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_CPU.mkdir(parents=True, exist_ok=True)
    # Make queue dirs world-writable so non-root prep pods (e.g. RAPIDS uid=1001) can rename files
    OUTPUT_PATH_GPU.chmod(0o777)
    OUTPUT_PATH_CPU.chmod(0o777)

    # Load distributions
    if KAGGLE_SEED_PATH and Path(KAGGLE_SEED_PATH).exists():
        dist = load_seed_distributions(KAGGLE_SEED_PATH)
    else:
        log.warning("[WARN] KAGGLE_SEED_PATH not set or not found — using hardcoded defaults")
        dist = _HARDCODED_DEFAULTS.copy()

    # Build fixed identity pools and inject into dist so all workers share them.
    cc_num_pool, merchant_pool = _build_identity_pools()
    dist["cc_num_pool"] = cc_num_pool
    dist["merchant_pool"] = merchant_pool
    log.info("[INFO] Identity pools: %d users, %d merchants", NUM_USERS, NUM_MERCHANTS)

    num_workers = NUM_WORKERS
    chunk_size = CHUNK_SIZE

    if STRESS_MODE:
        num_workers = min(num_workers * 4, os.cpu_count() or 8)
        chunk_size = chunk_size * 2
        log.info("[INFO] STRESS_MODE=true: workers=%d, chunk_size=%d", num_workers, chunk_size)

    log.info("[INFO] Starting data-gather: mode=%s workers=%d chunk_size=%d target_rows_per_sec=%d",
             RUN_MODE, num_workers, chunk_size, TARGET_ROWS_PER_SEC)

    # Rate governor: seconds each chunk-slot should occupy.
    # Asymmetric jitter: triangular(0.55, 0.92, 1.15) — biased low to mimic
    # real-world bursty patterns (occasional slow periods, rare brief spikes).
    target_chunk_time = (chunk_size / TARGET_ROWS_PER_SEC) if TARGET_ROWS_PER_SEC > 0 else 0.0

    total_rows = 0
    total_bytes = 0.0
    files_written = 0
    chunks_since_disk_check = 0
    start_time = time.time()
    last_telemetry = start_time
    rows_at_last_telemetry = 0
    actual_fraud_rate = dist.get("fraud_rate", FRAUD_RATE)

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(dist,),
    ) as pool:

        if RUN_MODE == "once":
            n_chunks = (TARGET_ROWS + chunk_size - 1) // chunk_size
            seed_offset = int(time.time()) & 0xFFFFFF
            chunk_args = [
                (i, min(chunk_size, TARGET_ROWS - i * chunk_size), seed_offset)
                for i in range(n_chunks)
            ]

            for df_chunk, _ in pool.imap_unordered(generate_chunk, chunk_args):
                if _SHUTDOWN:
                    break

                chunk_start = time.time()
                chunks_since_disk_check += 1
                if chunks_since_disk_check >= 10:
                    chunks_since_disk_check = 0
                    usage_pct, should_stop = check_disk_space(OUTPUT_PATH_GPU)
                    if should_stop:
                        sys.stdout.write(
                            f"[TELEMETRY] stage=gather status=PAUSED reason=disk_full "
                            f"disk_usage_pct={usage_pct * 100:.1f}\n"
                        )
                        sys.stdout.flush()
                        log.error("[ERROR] Disk usage %.0f%% > 95%%, stopping", usage_pct * 100)
                        break
                    if usage_pct > 0.8:
                        log.warning("[WARN] Disk usage %.0f%% > 80%%, consider cleanup", usage_pct * 100)

                idx = files_written
                fname = f"raw_chunk_{idx:06d}.parquet"
                df_chunk["chunk_ts"] = np.float64(time.time())  # pipeline latency anchor
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                out_gpu = OUTPUT_PATH_GPU / fname
                out_cpu = OUTPUT_PATH_CPU / fname
                pq.write_table(table, str(out_gpu), compression=None)
                pq.write_table(table, str(out_cpu), compression=None)

                total_rows += len(df_chunk)
                total_bytes += out_gpu.stat().st_size * 2
                files_written += 1
                actual_fraud_rate = float(df_chunk["is_fraud"].mean())

                now = time.time()
                if now - last_telemetry >= 1.0:
                    emit_telemetry(total_rows, total_bytes, files_written,
                                   num_workers, actual_fraud_rate, start_time,
                                   rows_since_last=total_rows - rows_at_last_telemetry,
                                   elapsed_since_last=now - last_telemetry)
                    rows_at_last_telemetry = total_rows
                    last_telemetry = now

                # Rate governor: sleep to hit target rate with asymmetric jitter.
                # triangular(0.55, 0.92, 1.15): biased slightly below target —
                # occasional dips to 55%, rare bursts to 115%, typical ~92%.
                if target_chunk_time > 0:
                    jitter = np.random.triangular(0.55, 0.92, 1.15)
                    sleep_for = target_chunk_time * jitter - (time.time() - chunk_start)
                    if sleep_for > 0:
                        time.sleep(sleep_for)

        else:
            # Continuous mode — loop, hot-reload stress config
            chunk_id = 0
            seed_offset = int(time.time()) & 0xFFFFFF

            while not _SHUTDOWN:
                Path("/.healthy").touch()  # liveness heartbeat
                # Hot-reload stress config
                sc = load_stress_config()
                if sc:
                    new_workers = int(sc.get("NUM_WORKERS", num_workers))
                    new_chunk = int(sc.get("CHUNK_SIZE", chunk_size))
                    if new_workers != num_workers or new_chunk != chunk_size:
                        num_workers = new_workers
                        chunk_size = new_chunk
                        log.info("[INFO] Stress config reloaded: workers=%d chunk_size=%d", num_workers, chunk_size)

                batch_args = [
                    (chunk_id + i, chunk_size, seed_offset)
                    for i in range(num_workers)
                ]
                chunk_id += num_workers

                for df_chunk, _ in pool.imap_unordered(generate_chunk, batch_args):
                    if _SHUTDOWN:
                        break

                    chunk_start = time.time()
                    chunks_since_disk_check += 1
                    if chunks_since_disk_check >= 10:
                        chunks_since_disk_check = 0
                        usage_pct, should_stop = check_disk_space(OUTPUT_PATH_GPU)
                        if should_stop:
                            sys.stdout.write(
                                f"[TELEMETRY] stage=gather status=PAUSED reason=disk_full "
                                f"disk_usage_pct={usage_pct * 100:.1f}\n"
                            )
                            sys.stdout.flush()
                            log.warning("[WARN] Disk full — pausing 60s")
                            time.sleep(60)
                            break
                        if usage_pct > 0.8:
                            log.warning("[WARN] Disk usage %.0f%% > 80%%", usage_pct * 100)

                    idx = files_written
                    fname = f"raw_chunk_{idx:06d}.parquet"
                    df_chunk["chunk_ts"] = np.float64(time.time())  # pipeline latency anchor
                    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                    out_gpu = OUTPUT_PATH_GPU / fname
                    out_cpu = OUTPUT_PATH_CPU / fname
                    pq.write_table(table, str(out_gpu), compression=None)
                    pq.write_table(table, str(out_cpu), compression=None)

                    total_rows += len(df_chunk)
                    try:
                        total_bytes += out_gpu.stat().st_size * 2
                    except FileNotFoundError:
                        pass  # prep worker already claimed the file — size estimate not critical
                    files_written += 1
                    actual_fraud_rate = float(df_chunk["is_fraud"].mean())

                    now = time.time()
                    if now - last_telemetry >= 1.0:
                        emit_telemetry(total_rows, total_bytes, files_written,
                                       num_workers, actual_fraud_rate, start_time,
                                       rows_since_last=total_rows - rows_at_last_telemetry,
                                       elapsed_since_last=now - last_telemetry)
                        rows_at_last_telemetry = total_rows
                        last_telemetry = now

                    if target_chunk_time > 0:
                        jitter = np.random.triangular(0.55, 0.92, 1.15)
                        sleep_for = target_chunk_time * jitter - (time.time() - chunk_start)
                        if sleep_for > 0:
                            time.sleep(sleep_for)

                if not _SHUTDOWN:
                    time.sleep(0.05)

    emit_telemetry(total_rows, total_bytes, files_written, num_workers, actual_fraud_rate, start_time)
    log.info("[INFO] Data gather complete: %d rows, %d files, %.1fs", total_rows, files_written, time.time() - start_time)


if __name__ == "__main__":
    main()
