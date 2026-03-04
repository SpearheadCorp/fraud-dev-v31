"""
Pod 2b: Data Prep (CPU-only)
Reads raw Parquet files, engineers 21 features using pandas/numpy only (no GPU).
Writes temporally-split feature files to OUTPUT_PATH for CPU vs GPU comparison.
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
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/raw"))
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
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame) -> tuple:
    df = df.sort_values("unix_time").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_PATH.glob("*.parquet"))
    if not input_files:
        log.error("[ERROR] No Parquet files found in %s", INPUT_PATH)
        sys.exit(1)

    log.info("[INFO] Reading %d Parquet files from %s", len(input_files), INPUT_PATH)
    t_read_start = time.perf_counter()
    df = pd.concat(
        [pd.read_parquet(str(f)) for f in input_files],
        ignore_index=True,
    )
    read_time = time.perf_counter() - t_read_start
    log.info("[INFO] Read %d rows in %.2fs", len(df), read_time)

    if len(df) == 0:
        log.error("[ERROR] No rows loaded from input files — aborting")
        sys.exit(1)
    if len(df) < 10000:
        log.warning("[WARN] Only %d rows loaded — feature quality and metrics may be poor", len(df))

    required_cols = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
    before = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < before:
        log.warning("[WARN] Dropped %d rows with NaN in required columns", before - len(df))

    df["merch_zipcode"] = df["merch_zipcode"].fillna(0.0)
    df["category"] = df["category"].fillna("misc_net")
    df["state"] = df["state"].fillna("CA")
    df["gender"] = df["gender"].fillna("F")

    log.info("[INFO] Running CPU feature engineering...")
    result, timing = engineer_features(df)
    log.info("[INFO] CPU complete: %.2fs total", timing["total"])

    log.info("[INFO] Performing temporal split (70/15/15)...")
    train, val, test = temporal_split(result)
    log.info(
        "[INFO] Split: train=%d val=%d test=%d (fraud: %.4f / %.4f / %.4f)",
        len(train), len(val), len(test),
        train["is_fraud"].mean(), val["is_fraud"].mean(), test["is_fraud"].mean(),
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_file = OUTPUT_PATH / f"features_{split_name}.parquet"
        table = pa.Table.from_pandas(split_df, preserve_index=False)
        pq.write_table(table, str(out_file), compression="snappy")
        log.info("[INFO] Wrote %s: %d rows (%.1f MB)", out_file.name, len(split_df), out_file.stat().st_size / 1e6)

    output_size_mb = sum(
        (OUTPUT_PATH / f"features_{s}.parquet").stat().st_size for s in ("train", "val", "test")
    ) / 1e6

    sys.stdout.write(
        f"[TELEMETRY] stage=prep-cpu files_read={len(input_files)} "
        f"rows_processed={len(df)} "
        f"cpu_time_s={timing['total']:.1f} "
        f"output_size_mb={output_size_mb:.0f}\n"
    )
    sys.stdout.flush()
    log.info("[INFO] CPU data prep complete")


if __name__ == "__main__":
    main()
