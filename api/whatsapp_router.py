"""
WhatsApp Alerts Router.

Exposes the Twilio inbound-message webhook (opt-in / location capture /
opt-out conversation flow) and a small status endpoint for checking
subscriber counts.

Twilio setup (one-time):
1. Create a free Twilio account and enable the WhatsApp Sandbox
   (or a production WhatsApp Business sender once approved).
2. Set environment variables: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
   TWILIO_WHATSAPP_FROM (e.g. "whatsapp:+14155238886" for the sandbox).
3. In the Twilio Console, set the Sandbox/Sender's "When a message comes in"
   webhook to: https://<your-api-host>/api/v1/whatsapp/webhook (HTTP POST).
4. Users text your Twilio WhatsApp number to join, then share their location
   when asked. That's the whole opt-in flow -- no separate app or form.

Without those env vars set, sends run in dry-run mode (logged, not sent) so
local development still works end-to-end.
"""

import logging
from fastapi import APIRouter, Request, Response

from .alerts_whatsapp import handle_inbound_message, get_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp Cyclone Alerts"])


@router.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Twilio inbound WhatsApp message webhook. Twilio posts form-encoded data;
    when the user shares a location pin, Latitude/Longitude fields are included.
    Responds with TwiML so Twilio relays the reply back to the user.
    """
    form = await request.form()
    from_phone = form.get("From", "")  # e.g. "whatsapp:+91xxxxxxxxxx"
    body = form.get("Body", "")
    latitude = form.get("Latitude")
    longitude = form.get("Longitude")

    reply_text = handle_inbound_message(
        from_phone=from_phone,
        body=body,
        latitude=float(latitude) if latitude else None,
        longitude=float(longitude) if longitude else None,
    )

    twiml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{_xml_escape(reply_text)}</Message></Response>"
    return Response(content=twiml, media_type="application/xml")


@router.get("/subscribers/count")
async def subscriber_count():
    """Quick visibility into subscriber counts (active / pending / opted out)."""
    return get_store().count()


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )