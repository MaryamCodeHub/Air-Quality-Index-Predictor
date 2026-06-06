# Hopsworks Integration Audit Report

**Generated:** May 29, 2026  
**Project:** AQI Intelligent Forecasting & Health Advisory System  
**Scope:** Full repository analysis of Hopsworks usage

---

## Executive Summary

**VERDICT: PARTIALLY IMPLEMENTED**

Hopsworks integration is **architecturally planned but not actually functional** in production. The codebase contains:
- ✅ **Design layer:** Complete connector implementation
- ✅ **Configuration:** Hopsworks settings in config.yaml
- ✅ **Workflow calls:** GitHub Actions workflow configured to push features
- ❌ **SDK installation:** hopsworks package NOT installed in environment
- ❌ **API key:** HOPSWORKS_API_KEY not configured
- ❌ **Data flow:** Features never actually reach Hopsworks (fallback to Parquet only)
- ❌ **Training loop:** Models train from Parquet, not from feature store
- ❌ **Model registry:** Models saved locally, not in Hopsworks

---

## Detailed Findings

### 1. Hopsworks Python SDK Installation

**Status:** ❌ **NOT INSTALLED**

**Evidence:**
```
✗ hopsworks SDK NOT installed
```

**Location:** `requirements.txt` line 11
```yaml
hopsworks>=3.0.0
```

**Problem:** 
- Package is listed in requirements.txt but has never been installed into the .venv
- All code that tries to `import hopsworks` will fail silently (caught in try/except)
- Fallback to local Parquet occurs automatically

**What needs to happen:**
```bash
pip install hopsworks>=3.0.0
```

---

### 2. Authentication & API Keys

**Status:** ❌ **NOT CONFIGURED**

**Evidence:**
```
✗ HOPSWORKS_API_KEY NOT set in environment
Connector status: {'connected': False, 'project_name': None, 'has_api_key': False}
```

**Location:** Multiple files reference this:
- [.github/workflows/hourly-ingest.yml](hourly-ingest.yml#L46): Tries to use `${{ secrets.HOPSWORKS_API_KEY }}`
- [.env](.env): HOPSWORKS_API_KEY not present
- [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py#L51-52): Checks for `os.getenv("HOPSWORKS_API_KEY")`

**Current behavior:**
```python
# From hopsworks_connector.py line 51-52
if not self.api_key:
    logger.warning("HOPSWORKS_API_KEY not found in .env. Feature store integration disabled...")
    return
```

**Configuration needed:**
1. Create account at https://hopsworks.ai (free tier available)
2. Create project: "aqi_forecasting"
3. Generate API key from project settings
4. Add to GitHub Secrets as `HOPSWORKS_API_KEY`
5. Optionally add to local `.env` for testing

---

### 3. Feature Store Creation/Access

**Status:** ⚠️ **DESIGNED BUT NOT FUNCTIONAL**

**Connector Implementation:** [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py)

**What EXISTS:**
```python
def _connect(self):
    """Connect to Hopsworks project and feature store."""
    import hopsworks
    project = hopsworks.login(
        host=self.host,
        api_key_value=self.api_key,
        project=self.project_name
    )
    self.fs = project.get_feature_store()
```

**Configuration in [config/config.yaml](config/config.yaml#L46-50):**
```yaml
hopsworks:
  project_name: "aqi_forecasting"
  host: "https://us-east-1.app.hopsworks.ai"
```

**What DOESN'T WORK:**
- Cannot execute `hopsworks.login()` because SDK not installed
- Cannot get feature store reference because authentication fails silently
- No actual feature groups created in Hopsworks

---

### 4. Feature Ingestion (Push to Hopsworks)

**Status:** ⚠️ **DESIGNED BUT NOT FUNCTIONAL**

**Code Location:** [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py#L87-145) - `push_features()` method

**Method signature:**
```python
def push_features(
    self,
    df: pd.DataFrame,
    feature_group_name: str = "aqi_features_24h",
    version: int = 1,
    primary_key: Optional[List[str]] = None,
    event_time: str = "timestamp"
) -> bool:
```

**Called from:** [run.py](run.py#L51-100) - `cmd_features()` function

**Workflow integration:** [.github/workflows/hourly-ingest.yml](hourly-ingest.yml#L45-49)
```yaml
- name: Engineer and push features
  env:
    HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
  run: |
    python run.py features
```

**Current behavior when executed:**
```
2026-05-29 23:19:50,460 | feature_store.hopsworks | WARNING | 
HOPSWORKS_API_KEY not found in .env. Feature store integration disabled. 
Features will use Parquet fallback.
```

**What SHOULD happen but DOESN'T:**
1. ✅ Load processed_aqi_data.parquet (WORKS)
2. ✅ Create feature group "aqi_features_24h" in Hopsworks (DOESN'T WORK - API key missing)
3. ✅ Insert features with primary_key=["city", "timestamp"] (DOESN'T WORK)
4. ✅ Enable online materialization (DOESN'T WORK)

**Actual current behavior:**
- Parquet file remains at `data/processed/processed_aqi_data.parquet`
- Nothing is pushed to Hopsworks
- No error is raised (graceful fallback)

---

### 5. Feature Materialization

**Status:** ⚠️ **PARTIALLY IMPLEMENTED**

**What IS implemented:**
- Hopsworks online store materialization code: [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py#L126)
```python
fg.insert(df, write_options={"start_offline_materialization": True})
```

**What IS NOT implemented:**
- No offline → online materialization scheduling
- No separate materialization command (the word "materialize" in CLI is actually an alias for "features", not true materialization)

**CLI alias:** [run.py](run.py#L102-103)
```python
def cmd_materialize(config):
    """Push features to Hopsworks (alias for cmd_features for backward compatibility)."""
    cmd_features(config)
```

**This is misleading** because:
- `python run.py materialize` just calls `python run.py features`
- True Hopsworks materialization (offline → online) would happen automatically during insert
- But since insert never happens (API key missing), materialization never occurs

---

### 6. Feature Retrieval for Training

**Status:** ⚠️ **DESIGNED BUT NOT FUNCTIONAL**

**Code Location:** [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py#L147-207) - `get_features()` method

**Called from:** [src/training/trainer.py](src/training/trainer.py#L108-142)

**In trainer.py:**
```python
# Lines 108-142
try:
    from src.feature_store import HopsworksConnector
    connector = HopsworksConnector(config)
    status = connector.get_feature_store_status()
    
    if status["connected"]:
        logger.info("Fetching historical features from Hopsworks feature store...")
        hw_df = connector.get_features(
            feature_names=["aqi", "pm25", "pm10", ...],
            feature_group_name="aqi_features_24h",
            version=1
        )
        if hw_df is not None and not hw_df.empty:
            df = hw_df
            logger.info("✓ Successfully fetched features from Hopsworks")
        else:
            logger.warning("No features returned from Hopsworks, using Parquet")
    else:
        logger.warning(f"Hopsworks not connected: {status}. Using Parquet fallback.")
except Exception as exc:
    logger.warning(f"Failed to fetch features: {exc}. Using Parquet fallback.")
```

**What happens in reality:**
1. `connector.get_feature_store_status()` returns `{'connected': False, ...}`
2. Falls back to: `logger.warning("Hopsworks not connected")`
3. ✅ Continues training with local Parquet: `df = pd.read_parquet(proc_path)`
4. ❌ Never reads features from Hopsworks

**Training data source (actual):** `data/processed/processed_aqi_data.parquet` (local)

---

### 7. Online Feature Retrieval for Real-time Predictions

**Status:** ❌ **NOT IMPLEMENTED**

**Code Location:** [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py#L209-246) - `get_online_features()` method

**Issue:** 
- Method exists but is never called from anywhere in the codebase
- Real-time prediction API would need this to serve fresh features
- Currently, prediction API would fail for real-time serving

**Search result:**
```
0 matches for "get_online_features" in entire repository (excluding this file)
```

---

### 8. Model Registry in Hopsworks

**Status:** ❌ **NOT IMPLEMENTED**

**What IS implemented:** Local model registry at [src/training/model_registry.py](src/training/model_registry.py)
- Saves trained models to: `models/{model_name}_{horizon}h.joblib`
- Saves metadata to: `models/{model_name}_{horizon}h.json`
- Marks best model: `models/best_{horizon}h.json`

**What IS NOT implemented:** Hopsworks Model Registry
- No code uses `project.get_model_registry()`
- No models registered in Hopsworks
- Models not versioned in Hopsworks
- Models not served through Hopsworks

**Current local artifacts:**
```
models/
├── best_72h.json
├── ridge_72h.joblib
├── ridge_72h.json
├── random_forest_72h.joblib
├── random_forest_72h.json
├── xgboost_72h.joblib
└── xgboost_72h.json
```

**All stored locally, not in Hopsworks.**

---

## File-by-File Hopsworks References

### Files that REFERENCE Hopsworks:

| File | Lines | References | Functional Status |
|------|-------|-----------|------------------|
| [.gitignore](.gitignore#L29) | 29 | Comment only | N/A |
| [.github/workflows/hourly-ingest.yml](hourly-ingest.yml#L46) | 46 | Environment variable | ❌ Key not set |
| [config/config.yaml](config/config.yaml#L46-50) | 46-50 | Configuration | ✅ Configured (but unused) |
| [run.py](run.py#L51-100) | 51-100 | cmd_features() function | ⚠️ Graceful fallback |
| [src/feature_store/hopsworks_connector.py](src/feature_store/hopsworks_connector.py) | 1-272 | Full connector | ❌ Not functional |
| [src/feature_store/__init__.py](src/feature_store/__init__.py#L3) | 3 | Import | ❌ Unused |
| [src/training/trainer.py](src/training/trainer.py#L108-142) | 108-142 | Feature retrieval attempt | ⚠️ Graceful fallback |
| [requirements.txt](requirements.txt#L11) | 11 | Dependency | ❌ Not installed |

### Files that DON'T reference Hopsworks (should):

| File | What's Missing |
|------|-----------------|
| [src/training/model_registry.py](src/training/model_registry.py) | No Hopsworks Model Registry integration |
| [src/api/routes.py](src/api/routes.py) | No online feature retrieval for real-time predictions |
| [src/dashboard/app.py](src/dashboard/app.py) | No Hopsworks feature monitoring |

---

## What's Actually Working vs. What's Fallback

### Current Data Flow (ACTUAL):

```
1. [AQICN API + OpenMeteo API] 
   ↓
2. data/raw/raw_aqi_data.parquet (LOCAL)
   ↓
3. Cleaning + Feature Engineering
   ↓
4. data/processed/processed_aqi_data.parquet (LOCAL)
   ↓
5. Training: Read from LOCAL Parquet
   ↓
6. Models saved to: models/*.joblib (LOCAL)
```

### Intended Data Flow (IF HOPSWORKS CONFIGURED):

```
1. [AQICN API + OpenMeteo API]
   ↓
2. data/raw/raw_aqi_data.parquet
   ↓
3. Cleaning + Feature Engineering
   ↓
4. data/processed/processed_aqi_data.parquet
   ↓
5. PUSH to Hopsworks Feature Store ← MISSING
   ↓
6. Training: READ from Hopsworks ← MISSING
   ↓
7. Models registered in Hopsworks Model Registry ← MISSING
```

---

## Checklist: What's Needed to Enable Hopsworks

- [ ] **1. Install SDK:** `pip install hopsworks>=3.0.0`
- [ ] **2. Create Hopsworks Account:** Sign up at https://hopsworks.ai (free tier)
- [ ] **3. Create Project:** "aqi_forecasting"
- [ ] **4. Generate API Key:** From project settings
- [ ] **5. Add to GitHub Secrets:** Setting name: `HOPSWORKS_API_KEY`
- [ ] **6. Add to local .env:** `HOPSWORKS_API_KEY=your_key_here` (for testing)
- [ ] **7. Test push:** `python run.py features`
- [ ] **8. Verify Feature Store:** Check Hopsworks UI for "aqi_features_24h" feature group
- [ ] **9. Integrate Model Registry:** Update `trainer.py` to save models to Hopsworks
- [ ] **10. Enable Real-time Serving:** Update API to use `get_online_features()`

---

## Project Requirements vs. Implementation

### Requirement 1: "Feature Store for accumulating historical data"

**Status:** ⚠️ **PARTIALLY MET**
- ✅ Can accumulate data (Parquet is a feature store)
- ❌ Not using cloud feature store (Hopsworks down)
- ⚠️ Limited to single machine (not scalable)

### Requirement 2: "Version control for features"

**Status:** ❌ **NOT MET**
- Features are appended to Parquet (no versioning)
- Hopsworks would provide versioning (not using it)

### Requirement 3: "Reproducible training data"

**Status:** ⚠️ **PARTIALLY MET**
- Parquet is immutable (good for reproducibility)
- Not using Hopsworks for version tracking (reduces confidence)

### Requirement 4: "Real-time feature serving"

**Status:** ❌ **NOT MET**
- No online feature store configured
- Predictions would need to compute features on-the-fly (slow)

### Requirement 5: "Model registry for deployment"

**Status:** ⚠️ **PARTIALLY MET**
- Local JSON + joblib registry works (single-machine only)
- Hopsworks Model Registry would enable multi-team collaboration (not using it)

---

## Impact Assessment

### What Works Today:

✅ Hourly data ingestion (AQICN + OpenMeteo APIs)  
✅ Feature engineering (8 → 53 features)  
✅ Model training (Ridge, RF, XGBoost for 72h)  
✅ Local model storage and versioning  
✅ Prediction API  
✅ Streamlit dashboard  

### What BREAKS if Hopsworks goes down:

Nothing. Everything is local.

### What WOULD BREAK if Hopsworks configured then network fails:

Training would fail (not graceful fallback anymore).

### What's MISSING from production readiness:

1. **Scalability:** Single-machine Parquet won't scale beyond 1 GB
2. **Collaboration:** No cloud feature store for team sharing
3. **Real-time serving:** No online feature cache for sub-100ms predictions
4. **Model versioning:** No Hopsworks Model Registry for team deployments
5. **Monitoring:** No Hopsworks data quality monitoring

---

## Code Quality Assessment

### Design: ⭐⭐⭐⭐⭐ (5/5)
- Clean connector abstraction
- Proper error handling with fallback
- Configuration-driven (not hardcoded)

### Implementation: ⭐⭐⭐☆☆ (3/5)
- Connector code is complete
- But SDK not installed (oversight)
- API key not configured (expected, needs manual setup)

### Integration: ⭐⭐☆☆☆ (2/5)
- Partially integrated into training
- Not integrated into serving (get_online_features unused)
- Not integrated into model serving

### Testing: ⭐⭐☆☆☆ (2/5)
- test_feast_integration.py exists (tests Feast, not Hopsworks)
- No Hopsworks-specific tests
- Would need Hopsworks API key to test

---

## Summary Table

| Component | Code Status | Installation Status | Configuration Status | Functional Status |
|-----------|------------|-------------------|-------------------|------------------|
| **SDK** | Listed in requirements.txt | ❌ Not installed | N/A | ❌ Non-functional |
| **Connector** | ✅ Fully implemented | ❌ SDK missing | ✅ Configured | ❌ Can't authenticate |
| **Feature Push** | ✅ Fully implemented | ❌ SDK missing | ❌ API key missing | ❌ Doesn't execute |
| **Feature Retrieval** | ✅ Fully implemented | ❌ SDK missing | ❌ API key missing | ⚠️ Fallback to Parquet |
| **Online Features** | ✅ Fully implemented | ❌ SDK missing | ❌ API key missing | ❌ Never called |
| **Model Registry (Hopsworks)** | ❌ Not implemented | N/A | N/A | ❌ Not implemented |
| **Local Parquet** | ✅ Implemented | ✅ Works | ✅ Works | ✅ Fully functional |
| **Local Model Registry** | ✅ Fully implemented | ✅ Works | ✅ Works | ✅ Fully functional |

---

## Final Verdict

### Overall Assessment: **PARTIALLY IMPLEMENTED**

**Hopsworks is:**
- ✅ **Well-designed** (clean architecture)
- ✅ **Well-configured** (config.yaml ready)
- ❌ **Not installed** (SDK missing)
- ❌ **Not authenticated** (API key missing)
- ❌ **Not functional** (graceful fallback to local Parquet)

**Project viability:**
- ✅ **Works today** (100% local, no dependencies on Hopsworks)
- ⚠️ **Will work without Hopsworks** (Parquet is sufficient for MVP)
- ⚠️ **Needs Hopsworks for production** (scalability, collaboration, real-time serving)

### Time to Enable Hopsworks: **~30 minutes**
1. Install SDK (2 min)
2. Create Hopsworks account (5 min)
3. Create project & API key (10 min)
4. Add to GitHub Secrets (3 min)
5. Test by running workflow (10 min)

### Recommendation:
- **For MVP/Development:** Keep using local Parquet (currently working)
- **For Production/Team:** Enable Hopsworks (requires 30 min setup)
- **For Real-time Serving:** Enable Hopsworks online store (additional 15 min)

---

**Report Generated:** May 29, 2026  
**Analysis Tool:** GitHub Copilot  
**Repository:** Air-Quality-Index-Predictor  
