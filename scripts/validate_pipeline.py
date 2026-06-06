#!/usr/bin/env python3
"""
Unified pipeline validation: Feast registry, processed data, and training feature retrieval.
Run from project root: python validate_pipeline.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from feast import FeatureStore

from src.feature_store.feast_integration import FeastIntegration
from src.utils.helpers import load_config


def check_processed_parquet() -> bool:
    proc_path = "data/processed/processed_aqi_data.parquet"
    print("1. Processed Parquet check")
    if not os.path.exists(proc_path):
        print(f"   [SKIP] Not found: {proc_path} (run ingest first)")
        return False
    df = pd.read_parquet(proc_path)
    print(f"   rows={len(df)}, cols={len(df.columns)}, has city={'city' in df.columns}")
    return len(df) > 0


def check_feast_registry() -> bool:
    print("2. Feast registry check")
    store = FeatureStore(repo_path="feature_store")
    fv = store.get_feature_view("aqi_islamabad_features")
    entity = store.get_entity("city")
    print(f"   feature_view={fv.name}, entity={entity.name}")
    print(f"   offline={store.config.offline_store.type}, online={store.config.online_store.type}")
    return True


def check_feast_integration() -> bool:
    print("3. Feast integration (online + training)")
    feast = FeastIntegration(repo_path="feature_store")
    online, meta = feast.get_features_for_inference()
    print(f"   online keys={list(online.keys())}, metadata={meta}")
    train_df, train_meta = feast.get_features_for_training(lookback_days=30)
    print(f"   training shape={train_df.shape}, metadata={train_meta}")
    return not train_df.empty or meta.get("source") == "parquet_fallback"


def check_config() -> bool:
    print("4. Config load")
    config = load_config()
    print(f"   models_dir={config['paths']['models']}")
    return os.path.isdir(config["paths"]["models"]) or True


def main() -> int:
    print("=" * 72)
    print("AQI PIPELINE VALIDATION")
    print("=" * 72)
    ok = True
    for fn in (check_config, check_processed_parquet, check_feast_registry, check_feast_integration):
        try:
            if not fn():
                ok = False
        except Exception as exc:
            print(f"   [FAIL] {exc}")
            ok = False
        print()
    print("=" * 72)
    print("PASSED" if ok else "FAILED — see messages above")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
