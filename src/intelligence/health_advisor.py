"""
Health Advisory Engine
=======================
AQI-based health recommendation system for Islamabad.

Maps AQI values to risk levels with actionable health advice.
"""

from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("intelligence.health_advisor")


class HealthAdvisor:
    """
    Rule-based health advisory engine using configurable AQI thresholds.

    Levels:
        0-50     Good              → Enjoy outdoors
        51-100   Moderate          → Sensitive groups caution
        101-150  Unhealthy (SG)    → Wear masks
        151-200  Unhealthy         → Avoid outdoor exertion
        201-300  Very Unhealthy    → Stay indoors
        301-500  Hazardous         → Health emergency
    """

    def __init__(self, config: Dict[str, Any]):
        self.levels = config["health_advisory"]["levels"]

    def get_advice(self, aqi: Optional[float]) -> Dict[str, Any]:
        """
        Get health advisory for a given AQI value.

        Args:
            aqi: Current or forecasted AQI value

        Returns:
            Dict with level, color, advice, and aqi value
        """
        if aqi is None:
            return {
                "aqi": None,
                "level": "Unknown",
                "color": "gray",
                "advice": "AQI data unavailable. Exercise caution.",
            }

        aqi = float(aqi)

        for level_config in self.levels:
            lo, hi = level_config["range"]
            if lo <= aqi <= hi:
                advisory = {
                    "aqi": round(aqi, 1),
                    "level": level_config["level"],
                    "color": level_config["color"],
                    "advice": level_config["advice"],
                }
                logger.info(f"AQI {aqi:.0f} → {advisory['level']}")
                return advisory

        # AQI above all defined ranges
        return {
            "aqi": round(aqi, 1),
            "level": "Hazardous",
            "color": "maroon",
            "advice": "Extreme hazard: Stay indoors with air filtration.",
        }

    def get_forecast_advisories(self, predictions: Dict[int, float]) -> Dict[int, Dict]:
        """
        Get advisories for multiple forecast horizons.

        Args:
            predictions: {horizon_hours: predicted_aqi}

        Returns:
            {horizon_hours: advisory_dict}
        """
        advisories = {}
        for horizon, aqi in predictions.items():
            advisories[horizon] = self.get_advice(aqi)
        return advisories
