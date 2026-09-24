from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import json
import requests


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MultiChannelAlertManager:

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None
    ):
        self.discord_url = discord_webhook_url
        self.slack_url = slack_webhook_url
        self.tg_token = telegram_bot_token
        self.tg_chat_id = telegram_chat_id

    def format_alert_payload(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[Dict] = None
    ) -> Dict:
        color = 3066993 if severity == AlertSeverity.INFO else (16776960 if severity == AlertSeverity.WARNING else 15158332)
        payload = {
            "title": f"[{severity.value}] {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        return payload

    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: Optional[Dict] = None
    ) -> Dict[str, bool]:
        payload = self.format_alert_payload(title, message, severity, metadata)
        delivery_status = {}

        if self.discord_url:
            discord_body = {
                "embeds": [{
                    "title": payload["title"],
                    "description": payload["description"],
                    "color": payload["color"],
                    "fields": [{"name": k, "value": str(v), "inline": True} for k, v in payload["metadata"].items()]
                }]
            }
            try:
                r = requests.post(self.discord_url, json=discord_body, timeout=5)
                delivery_status["discord"] = (r.status_code == 204 or r.status_code == 200)
            except Exception:
                delivery_status["discord"] = False

        if self.slack_url:
            slack_body = {
                "text": f"*{payload['title']}*\n{payload['description']}\n```json\n{json.dumps(payload['metadata'], indent=2)}\n```"
            }
            try:
                r = requests.post(self.slack_url, json=slack_body, timeout=5)
                delivery_status["slack"] = (r.status_code == 200)
            except Exception:
                delivery_status["slack"] = False

        return delivery_status
