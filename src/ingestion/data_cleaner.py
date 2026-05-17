"""
Data Ingestion Cleaner Proxy Module
====================================
Maintains original namespace imports for backward compatibility
while routing operations to the new decoupled cleaning layers.
"""

from typing import Any, Dict
import pandas as pd

from src.cleaning.cleaner import DataCleaner, run_cleaning_pipeline
