"""
Alert System — SMTP Email Notifications
=========================================
Sends email alerts when AQI exceeds hazardous thresholds.

Configured via config.yaml (alerting section).
Disabled by default — set alerting.enabled: true to activate.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.logger import setup_logger

logger = setup_logger("intelligence.alerting")


class AlertSystem:
    """SMTP-based email alerting for hazardous AQI conditions in Islamabad."""

    def __init__(self, config: Dict[str, Any]):
        alert_cfg = config["alerting"]
        self.enabled = alert_cfg.get("enabled", False)
        self.smtp_server = alert_cfg.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = alert_cfg.get("smtp_port", 587)
        self.sender_email = alert_cfg.get("sender_email", "")
        self.sender_password = alert_cfg.get("sender_password", "")
        self.recipients = alert_cfg.get("recipients", [])
        self.aqi_threshold = alert_cfg.get("aqi_alert_threshold", 200)

    def check_and_alert(self, aqi: Optional[float], advisory: Dict[str, Any]) -> bool:
        """
        Check if AQI exceeds threshold and send alert if enabled.

        Args:
            aqi: Current AQI value
            advisory: Health advisory dict from HealthAdvisor

        Returns:
            True if alert was sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Alerting disabled in config")
            return False

        if aqi is None or aqi < self.aqi_threshold:
            return False

        logger.warning(f"AQI {aqi:.0f} exceeds threshold {self.aqi_threshold} — sending alert")
        return self._send_email(aqi, advisory)

    def _send_email(self, aqi: float, advisory: Dict[str, Any]) -> bool:
        """Send the alert email via SMTP."""
        subject = f"⚠️ AQI ALERT — Islamabad AQI: {aqi:.0f} ({advisory['level']})"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
        <h2 style="color: {advisory.get('color', 'red')};">
            ⚠️ Air Quality Alert — Islamabad
        </h2>
        <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
            <tr style="background: #f5f5f5;">
                <td style="padding: 10px; font-weight: bold;">Current AQI</td>
                <td style="padding: 10px;">{aqi:.0f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Risk Level</td>
                <td style="padding: 10px;">{advisory['level']}</td>
            </tr>
            <tr style="background: #f5f5f5;">
                <td style="padding: 10px; font-weight: bold;">Advisory</td>
                <td style="padding: 10px;">{advisory['advice']}</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Timestamp</td>
                <td style="padding: 10px;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</td>
            </tr>
        </table>
        <p style="color: #666; font-size: 12px; margin-top: 20px;">
            AQI Intelligent Forecasting & Health Advisory System
        </p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText(body, "html"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            logger.info(f"Alert email sent to {len(self.recipients)} recipients")
            return True
        except Exception as exc:
            logger.error(f"Failed to send alert email: {exc}")
            return False
