"""
AQICN Client Module
====================
Fetches and standardizes air quality index records from the AQICN API.
Uses a dynamic mapping layer to keep ingestion API-agnostic.
"""

from typing import Any, Dict, Optional

from src.ingestion.base_client import BaseAPIClient
from src.ingestion.data_normalizer import DataNormalizer
from src.utils.logger import setup_logger

logger = setup_logger("ingestion.api_clients.aqicn")


class AQICNClient(BaseAPIClient):
    """
    Fetches real-time AQI + pollutant data for Islamabad from AQICN.
    API docs: https://aqicn.org/json-api/doc/
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "aqicn")
        aqicn = config["api"]["aqicn"]
        self.base_url = aqicn["base_url"]
        self.api_key = aqicn["api_key"]
        self.city_id = config["city"]["id"]
        self.normalizer = DataNormalizer(config)

    def fetch_aqi(self) -> Optional[Dict[str, Any]]:
        """Fetch current AQI for Islamabad. Falls back to cache on failure."""
        url = f"{self.base_url}/feed/{self.city_id}/"
        params = {"token": self.api_key}

        response = self._make_request(url, params)
        if response is None:
            logger.warning("[AQICN] Live request failed — trying cache fallback")
            response = self._get_cached_response(url)
            if response is None:
                logger.error("[AQICN] No cached data available")
                return None

        return self._parse_response(response)

    def _parse_response(self, response: Dict) -> Optional[Dict[str, Any]]:
        """Flatten nested AQICN JSON into standard internal format using config mappings."""
        if response.get("status") != "ok":
            logger.error(f"[AQICN] Bad status: {response.get('data')}")
            return None

        parsed = self.normalizer.normalize_aqicn(response)
        logger.info(f"[AQICN] Islamabad AQI = {parsed.get('aqi')}")
        return parsed
