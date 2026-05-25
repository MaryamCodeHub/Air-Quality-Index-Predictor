"""
Open-Meteo Weather Client Module
=================================
Fetches and standardizes weather records from the Open-Meteo API.
Implements the Open-Meteo integration under standard weather client namespaces.
"""

from typing import Any, Dict, Optional

from src.ingestion.base_client import BaseAPIClient
from src.ingestion.data_normalizer import DataNormalizer
from src.utils.logger import setup_logger

logger = setup_logger("ingestion.api_clients.open_meteo")


class OpenMeteoClient(BaseAPIClient):
    """
    Fetches current weather data for Islamabad from Open-Meteo.
    API docs: https://open-meteo.com/en/docs
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "open_meteo")
        om = config["api"]["open_meteo"]
        self.forecast_url = om["forecast_url"]
        self.archive_url = om["archive_url"]
        self.hourly_params = om["hourly_params"]
        self.lat = config["city"]["lat"]
        self.lon = config["city"]["lon"]
        self.normalizer = DataNormalizer(config)

    def fetch_weather(self) -> Optional[Dict[str, Any]]:
        """Fetch current weather for Islamabad. Falls back to cache on failure."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "hourly": self.hourly_params,
            "timezone": "Asia/Karachi",
            "forecast_days": 1,
        }

        response = self._make_request(self.forecast_url, params)
        if response is None:
            logger.warning("[OpenMeteo] Live request failed — trying cache fallback")
            response = self._get_cached_response(self.forecast_url)
            if response is None:
                logger.error("[OpenMeteo] No cached data available")
                return None

        return self._parse_response(response)

    def fetch_historical_weather(self, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """
        Fetch historical weather data for Islamabad from Open-Meteo archive.
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            
        Returns:
            Dict with hourly timeseries data (can contain thousands of rows)
        """
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.hourly_params,
            "timezone": "Asia/Karachi",
        }
        
        response = self._make_request(self.archive_url, params)
        if response is None:
            logger.error(f"[OpenMeteo] Failed to fetch historical data ({start_date} to {end_date})")
            return None
        
        logger.info(
            f"[OpenMeteo] Historical fetch succeeded: {start_date} to {end_date}"
        )
        return response

    def _parse_response(self, response: Dict) -> Optional[Dict[str, Any]]:
        """Extract hourly reading from Open-Meteo JSON using config mappings."""
        parsed = self.normalizer.normalize_open_meteo(response)
        if not parsed:
            logger.error("[OpenMeteo] Parsing response failed")
            return None

        logger.info(
            f"[OpenMeteo] Islamabad: {parsed.get('temperature')}°C, "
            f"humidity={parsed.get('humidity')}%"
        )
        return parsed
