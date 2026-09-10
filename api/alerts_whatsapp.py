"""
WhatsApp Cyclone Alert Service.

Public-facing broadcast alerts, modeled on how organizations like RBI push
WhatsApp safety advisories: a user opts in by messaging the business number,
shares their location once via WhatsApp's native location-share, and from
then on receives push alerts automatically whenever a cyclone is detected
near them -- no app, no dashboard, no repeated action required.

Components:
- SubscriberStore: Redis-backed persistence (phone -> lat/lon/basin/status)
  with an in-memory fallback so the service still works without Redis running,
  mirroring the pattern used in api/rate_limiter.py.
- TwilioWhatsAppClient: thin wrapper around the Twilio WhatsApp Business API.
  Runs in "dry run" mode (logs instead of sending) if credentials aren't
  configured, so the rest of the app never crashes because of missing keys.
- Inbound message handling: opt-in / location capture / opt-out via
  conversational WhatsApp messages.
- Outbound broadcasting: given a cyclone prediction result, finds subscribers
  within the alert radius of the storm's current center and messages them.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

# ==========================================
# Configuration (all overridable via environment variables)
# ==========================================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Twilio sandbox default

# Radius around a detected storm's eye within which subscribers get alerted.
ALERT_RADIUS_KM = float(os.getenv("WHATSAPP_ALERT_RADIUS_KM", "300"))

# Minimum IMD category (by ordinal position in regional_bounds.json's
# imd_categories list) required before an alert is sent at all, so people
# aren't paged for a harmless low-pressure area.
MIN_ALERT_CATEGORY_INDEX = int(os.getenv("WHATSAPP_MIN_ALERT_CATEGORY_INDEX", "2"))  # "CS" and above

# Don't re-alert the same subscriber for the same storm more than once
# within this cooldown window.
ALERT_COOLDOWN_SECONDS = int(os.getenv("WHATSAPP_ALERT_COOLDOWN_SECONDS", str(6 * 3600)))

SUBSCRIBERS_KEY = "whatsapp:subscribers"
ALERT_LOG_PREFIX = "whatsapp:alert_sent"
LOCAL_STORE_PATH = os.getenv("WHATSAPP_LOCAL_STORE_PATH", "data/whatsapp_subscribers.json")

OPT_IN_KEYWORDS = {"hi", "hello", "join", "subscribe", "start", "cyclone alert", "alert"}
OPT_OUT_KEYWORDS = {"stop", "unsubscribe", "cancel", "quit", "opt out"}


# ==========================================
# Distance helper
# ==========================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# ==========================================
# Subscriber persistence
# ==========================================
class SubscriberStore:
    """
    Stores WhatsApp subscribers keyed by phone number.

    Record shape:
    {
        "phone": "whatsapp:+9198xxxxxxx",
        "latitude": float | None,
        "longitude": float | None,
        "basin": str | None,
        "status": "pending_location" | "active" | "opted_out",
        "opted_in_at": float (unix ts),
        "updated_at": float (unix ts),
    }

    Uses Redis (REDIS_HOST/REDIS_PORT env vars, same as rate_limiter.py) when
    reachable. Falls back to a local JSON file so the opt-in flow still works
    in development without any infra running.
    """

    def __init__(self, redis_host: Optional[str] = None, redis_port: int = 6379):
        self.redis_client = None
        redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", redis_port))
        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True, socket_timeout=1
            )
            self.redis_client.ping()
            logger.info("WhatsApp subscriber store connected to Redis.")
        except Exception:
            self.redis_client = None
            logger.info("WhatsApp subscriber store: Redis unavailable, using local JSON file fallback.")
            os.makedirs(os.path.dirname(LOCAL_STORE_PATH) or ".", exist_ok=True)
            if not os.path.exists(LOCAL_STORE_PATH):
                with open(LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
                    json.dump({}, f)

    # ---- low level get/set of the whole subscriber map ----
    def _read_all(self) -> Dict[str, Dict[str, Any]]:
        if self.redis_client:
            raw = self.redis_client.hgetall(SUBSCRIBERS_KEY)
            return {k: json.loads(v) for k, v in raw.items()}
        with open(LOCAL_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_one(self, phone: str, record: Dict[str, Any]):
        if self.redis_client:
            self.redis_client.hset(SUBSCRIBERS_KEY, phone, json.dumps(record))
            return
        data = self._read_all()
        data[phone] = record
        with open(LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ---- public API ----
    def get(self, phone: str) -> Optional[Dict[str, Any]]:
        return self._read_all().get(phone)

    def upsert_opt_in(self, phone: str) -> Dict[str, Any]:
        existing = self.get(phone) or {}
        record = {
            "phone": phone,
            "latitude": existing.get("latitude"),
            "longitude": existing.get("longitude"),
            "basin": existing.get("basin"),
            "status": "pending_location" if not existing.get("latitude") else "active",
            "opted_in_at": existing.get("opted_in_at", time.time()),
            "updated_at": time.time(),
        }
        self._write_one(phone, record)
        return record

    def set_location(self, phone: str, latitude: float, longitude: float, basin: Optional[str] = None) -> Dict[str, Any]:
        existing = self.get(phone) or {"phone": phone, "opted_in_at": time.time()}
        existing.update({
            "latitude": latitude,
            "longitude": longitude,
            "basin": basin,
            "status": "active",
            "updated_at": time.time(),
        })
        self._write_one(phone, existing)
        return existing

    def opt_out(self, phone: str) -> Dict[str, Any]:
        existing = self.get(phone) or {"phone": phone}
        existing["status"] = "opted_out"
        existing["updated_at"] = time.time()
        self._write_one(phone, existing)
        return existing

    def all_active(self) -> List[Dict[str, Any]]:
        return [r for r in self._read_all().values() if r.get("status") == "active" and r.get("latitude") is not None]

    def count(self) -> Dict[str, int]:
        all_records = list(self._read_all().values())
        active = sum(1 for r in all_records if r.get("status") == "active")
        pending = sum(1 for r in all_records if r.get("status") == "pending_location")
        opted_out = sum(1 for r in all_records if r.get("status") == "opted_out")
        return {"active": active, "pending_location": pending, "opted_out": opted_out, "total": len(all_records)}

    # ---- alert de-duplication ----
    def already_alerted(self, phone: str, storm_id: str) -> bool:
        key = f"{ALERT_LOG_PREFIX}:{storm_id}:{phone}"
        if self.redis_client:
            return self.redis_client.exists(key) == 1
        # Local fallback: piggyback on the subscriber record itself.
        rec = self.get(phone) or {}
        last_alerts = rec.get("last_alerts", {})
        sent_at = last_alerts.get(storm_id)
        return bool(sent_at and (time.time() - sent_at) < ALERT_COOLDOWN_SECONDS)

    def mark_alerted(self, phone: str, storm_id: str):
        if self.redis_client:
            key = f"{ALERT_LOG_PREFIX}:{storm_id}:{phone}"
            self.redis_client.setex(key, ALERT_COOLDOWN_SECONDS, "1")
            return
        rec = self.get(phone) or {"phone": phone}
        rec.setdefault("last_alerts", {})[storm_id] = time.time()
        self._write_one(phone, rec)


# ==========================================
# Twilio WhatsApp send client
# ==========================================
class TwilioWhatsAppClient:
    """
    Thin wrapper around Twilio's WhatsApp Business API.

    If TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN aren't set, this runs in dry-run
    mode: messages are logged, not sent, so the rest of the system keeps
    working during development or if Twilio isn't configured yet.
    """

    def __init__(self):
        self.dry_run = not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
        self._client = None
        if not self.dry_run:
            try:
                from twilio.rest import Client
                self._client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            except Exception as e:
                logger.warning(f"Twilio client init failed ({e}); falling back to dry-run mode.")
                self.dry_run = True

    def send(self, to_phone: str, body: str) -> bool:
        to = to_phone if to_phone.startswith("whatsapp:") else f"whatsapp:{to_phone}"
        if self.dry_run:
            logger.info(f"[DRY RUN - no Twilio credentials configured] Would send to {to}: {body}")
            return True
        try:
            self._client.messages.create(from_=TWILIO_WHATSAPP_FROM, to=to, body=body)
            return True
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message to {to}: {e}")
            return False


# ==========================================
# Shared singletons
# ==========================================
_store: Optional[SubscriberStore] = None
_twilio: Optional[TwilioWhatsAppClient] = None


def get_store() -> SubscriberStore:
    global _store
    if _store is None:
        _store = SubscriberStore()
    return _store


def get_twilio_client() -> TwilioWhatsAppClient:
    global _twilio
    if _twilio is None:
        _twilio = TwilioWhatsAppClient()
    return _twilio


# ==========================================
# Inbound message handling (opt-in / location / opt-out)
# ==========================================
def handle_inbound_message(
    from_phone: str,
    body: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """
    Processes one inbound WhatsApp message and returns the reply text to send back.
    `latitude`/`longitude` are populated when the user shares a location pin
    (Twilio passes these as top-level form fields on the webhook payload).
    """
    store = get_store()
    text = (body or "").strip().lower()

    if latitude is not None and longitude is not None:
        basin = _dispatch_basin_id(latitude, longitude)
        store.set_location(from_phone, latitude, longitude, basin)
        return (
            "📍 Location saved. You're now subscribed to cyclone alerts for your area.\n\n"
            "You'll get a WhatsApp message automatically if a cyclone is detected near you. "
            "Reply STOP anytime to unsubscribe."
        )

    if any(text == kw or text.startswith(kw) for kw in OPT_OUT_KEYWORDS):
        store.opt_out(from_phone)
        return "You've been unsubscribed from cyclone alerts. Send HI anytime to rejoin."

    if any(text == kw or kw in text for kw in OPT_IN_KEYWORDS) or store.get(from_phone) is None:
        store.upsert_opt_in(from_phone)
        return (
            "🌀 Welcome to Cyclone Alerts.\n\n"
            "To start receiving warnings for your area, please share your location: "
            "tap the ➕ / attach icon in WhatsApp → Location → Send your current location.\n\n"
            "Reply STOP anytime to unsubscribe."
        )

    existing = store.get(from_phone)
    if existing and existing.get("status") == "pending_location":
        return "Still waiting on your location 📍 — tap the attach icon → Location → Send your current location to finish subscribing."

    return "You're already subscribed to cyclone alerts for your area. Reply STOP to unsubscribe."


def _dispatch_basin_id(lat: float, lon: float) -> Optional[str]:
    try:
        with open("configs/regional_bounds.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        for basin_key, basin_data in config.get("basins", {}).items():
            b = basin_data.get("bounds", {})
            if b.get("min_lat", -90) <= lat <= b.get("max_lat", 90) and b.get("min_lon", -180) <= lon <= b.get("max_lon", 180):
                return basin_key
        return config.get("default_fallback_basin")
    except Exception:
        return None


# ==========================================
# Outbound broadcast: cyclone detected -> alert nearby subscribers
# ==========================================
def _category_index(category_code: str) -> int:
    try:
        with open("configs/regional_bounds.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        codes = [c["code"] for c in config.get("imd_categories", [])]
        return codes.index(category_code) if category_code in codes else 0
    except Exception:
        return 0


def build_alert_message(prediction: Dict[str, Any], distance_km: float) -> str:
    intensity = prediction.get("intensity_classification", {})
    basin = prediction.get("ocean_basin", {})
    eye = prediction.get("eye_detection", {})
    category = intensity.get("category_code", "?")
    wind_kmph = intensity.get("maximum_sustained_wind_kmph", 0)
    pressure = intensity.get("central_pressure_hpa", 0)

    return (
        f"⚠️ CYCLONE ALERT ({basin.get('name', 'your region')})\n\n"
        f"A tropical system (category {category}) has been detected roughly "
        f"{int(distance_km)} km from your saved location.\n"
        f"Max sustained winds: {wind_kmph:.0f} km/h | Central pressure: {pressure:.0f} hPa\n"
        f"Center: {eye.get('latitude'):.1f}°N, {eye.get('longitude'):.1f}°E\n\n"
        "Please follow local IMD/authority advisories and avoid coastal areas until further notice.\n"
        "Reply STOP to unsubscribe from these alerts."
    )


def broadcast_alert_for_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a full-pipeline prediction result, finds subscribers within
    ALERT_RADIUS_KM of the detected storm's eye and sends them a WhatsApp
    alert (skipping anyone already alerted for this storm within the
    cooldown window, and skipping low-intensity systems below the
    configured minimum category).

    Returns a small summary dict for logging; never raises.
    """
    summary = {"eligible": False, "candidates": 0, "sent": 0, "skipped_cooldown": 0, "failed": 0}
    try:
        intensity = prediction.get("intensity_classification", {})
        eye = prediction.get("eye_detection", {})
        storm_id = prediction.get("storm_id", "unknown_storm")
        category = intensity.get("category_code", "")
        lat, lon = eye.get("latitude"), eye.get("longitude")

        if lat is None or lon is None:
            return summary
        if _category_index(category) < MIN_ALERT_CATEGORY_INDEX:
            logger.info(f"Storm {storm_id} is category {category}, below alert threshold; no WhatsApp alerts sent.")
            return summary

        summary["eligible"] = True
        store = get_store()
        twilio = get_twilio_client()

        for sub in store.all_active():
            dist = haversine_km(lat, lon, sub["latitude"], sub["longitude"])
            if dist > ALERT_RADIUS_KM:
                continue
            summary["candidates"] += 1
            phone = sub["phone"]
            if store.already_alerted(phone, storm_id):
                summary["skipped_cooldown"] += 1
                continue
            message = build_alert_message(prediction, dist)
            if twilio.send(phone, message):
                store.mark_alerted(phone, storm_id)
                summary["sent"] += 1
            else:
                summary["failed"] += 1

        logger.info(f"WhatsApp broadcast for {storm_id}: {summary}")
        return summary
    except Exception as e:
        logger.error(f"WhatsApp broadcast failed: {e}")
        return summary