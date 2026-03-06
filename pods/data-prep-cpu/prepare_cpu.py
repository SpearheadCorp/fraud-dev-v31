"""
Pod: data-prep-cpu
Continuous file-queue worker (CPU-only). Atomically claims raw parquet chunks from
INPUT_PATH, engineers 21 features via pandas/numpy, writes to OUTPUT_PATH.
Multiple replicas race-safely share the queue via POSIX rename atomicity.
"""
import os
import sys
import time
import logging
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    log.info("[INFO] Signal %s received — shutting down", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/raw/cpu"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/features-cpu"))

# ---------------------------------------------------------------------------
# Category / state maps (global)
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
    "health_fitness", "travel", "kids_pets", "home",
]
CATEGORY_MAP = {cat: idx for idx, cat in enumerate(ALL_CATEGORIES)}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]
STATE_MAP = {s: idx for idx, s in enumerate(US_STATES)}

FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
    "is_fraud",
]


# ---------------------------------------------------------------------------
# Haversine distance (numpy)
# ---------------------------------------------------------------------------

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Feature engineering (CPU-only)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> tuple:
    """Run full feature engineering on CPU (pandas/numpy). Returns (df_features, timing_dict)."""
    t = {}
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    amt_log = np.log1p(df["amt"].values)
    amt_mean = df["amt"].mean()
    amt_std = df["amt"].std()
    amt_scaled = (df["amt"].values - amt_mean) / max(amt_std, 1e-9)
    t["amount"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    dt = pd.to_datetime(df["unix_time"], unit="s")
    hour_of_day = dt.dt.hour.astype(np.int8)
    day_of_week = dt.dt.dayofweek.astype(np.int8)
    is_weekend = (day_of_week >= 5).astype(np.int8)
    is_night = (hour_of_day <= 5).astype(np.int8)
    t["temporal"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    distance_km = haversine_np(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )
    t["distance"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    category_encoded = df["category"].map(CATEGORY_MAP).fillna(0).astype(np.int8)
    state_encoded = df["state"].map(STATE_MAP).fillna(0).astype(np.int8)
    gender_encoded = (df["gender"] == "F").astype(np.int8)
    t["encoding"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    city_pop_log = np.log1p(df["city_pop"].values)
    zip_region = (df["zip"].values // 10000).astype(np.int8)
    t["misc"] = time.perf_counter() - t1

    out = pd.DataFrame(
        {
            "amt_log": amt_log,
            "amt_scaled": amt_scaled,
            "hour_of_day": hour_of_day.values,
            "day_of_week": day_of_week.values,
            "is_weekend": is_weekend.values,
            "is_night": is_night.values,
            "distance_km": distance_km,
            "category_encoded": category_encoded.values,
            "state_encoded": state_encoded.values,
            "gender_encoded": gender_encoded.values,
            "city_pop_log": city_pop_log,
            "zip_region": zip_region,
            "amt": df["amt"].values,
            "lat": df["lat"].values,
            "long": df["long"].values,
            "city_pop": df["city_pop"].values,
            "unix_time": df["unix_time"].values,
            "merch_lat": df["merch_lat"].values,
            "merch_long": df["merch_long"].values,
            "merch_zipcode": df["merch_zipcode"].values,
            "zip": df["zip"].values,
            "is_fraud": df["is_fraud"].values,
        },
        index=df.index,
    )
    t["total"] = time.perf_counter() - t0
    return out, t


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(chunk_id: int, rows: int, cpu_time: float) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage=prep-cpu chunk_id={chunk_id} rows={rows} "
        f"cpu_time_s={cpu_time:.3f}\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main — continuous file-queue loop
# ---------------------------------------------------------------------------

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
_PASSTHROUGH_COLS = ["cc_num", "merchant", "trans_num", "is_fraud", "amt", "category"]

# Pod-unique prefix so multiple replicas don't overwrite each other's output files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


def main() -> None:
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] data-prep-cpu started: INPUT=%s OUTPUT=%s pod=%s",
             INPUT_PATH, OUTPUT_PATH, _POD_PREFIX)

    chunk_id = 0
    while not _SHUTDOWN:
        # --- Claim next available chunk via atomic rename ---
        files = sorted(f for f in INPUT_PATH.glob("*.parquet")
                       if not f.name.endswith((".processing", ".done")))
        claimed: Path | None = None
        for f in files:
            proc = Path(str(f) + ".processing")
            try:
                f.rename(proc)
                claimed = proc
                break
            except (FileNotFoundError, OSError):
                continue

        if claimed is None:
            time.sleep(0.5)
            continue

        # --- Load ---
        try:
            df = pd.read_parquet(str(claimed))
        except Exception as exc:
            log.warning("[WARN] Failed to read %s: %s — skipping", claimed.name, exc)
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        if len(df) == 0:
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        # --- Clean ---
        df["merch_zipcode"] = df["merch_zipcode"].fillna(0.0)
        df["category"] = df["category"].fillna("misc_net")
        df["state"] = df["state"].fillna("CA")
        df["gender"] = df["gender"].fillna("F")
        df = df.dropna(subset=_REQUIRED_COLS)
        if len(df) == 0:
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        # --- Feature engineering ---
        output, timing = engineer_features(df)

        # --- Carry forward graph-critical columns for scorer ---
        for col in _PASSTHROUGH_COLS:
            if col in df.columns and col not in output.columns:
                output[col] = df[col].values

        # --- Write output (atomic: write to .tmp then rename so scorer never sees partial file) ---
        out_file = OUTPUT_PATH / f"features_{_POD_PREFIX}_{chunk_id:06d}.parquet"
        tmp_file = out_file.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pandas(output, preserve_index=False), str(tmp_file))
        tmp_file.rename(out_file)
        claimed.rename(str(claimed).replace(".processing", ".done"))

        emit_telemetry(chunk_id=chunk_id, rows=len(df), cpu_time=timing["total"])
        log.info("[INFO] chunk %06d: %d rows cpu_time=%.3fs", chunk_id, len(df), timing["total"])
        chunk_id += 1

    log.info("[INFO] data-prep-cpu shutdown complete after %d chunks", chunk_id)


if __name__ == "__main__":
    main()
