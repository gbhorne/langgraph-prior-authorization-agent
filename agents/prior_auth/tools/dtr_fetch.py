"""
agents/prior_auth/tools/dtr_fetch.py
=====================================
PA-2: Documentation Templates and Rules (DTR) Questionnaire Fetcher

What this step does:
  1. Checks Firestore for a cached Questionnaire for {payer_id}:{cpt_code}
  2. If cached and not expired (24h TTL) → returns cached version
  3. If not cached or expired → fetches from payer's DTR FHIR endpoint
  4. Stores fetched Questionnaire in Firestore for future requests

Why caching matters:
  - Payer questionnaires change infrequently (quarterly at most)
  - Repeated payer API calls for the same CPT code waste time
  - Firestore cache serves questionnaires in milliseconds vs seconds

Da Vinci DTR flow:
  - Payer exposes a FHIR Questionnaire endpoint per service category
  - We fetch the Questionnaire, then auto-populate it using FHIR data (PA-3)
  - The populated QuestionnaireResponse goes into the PAS Bundle (PA-4)

Reference: https://build.fhir.org/ig/HL7/davinci-dtr/
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import aiohttp
from google.cloud import firestore

from shared.config import CDSSConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(payer_id: str, cpt_code: str) -> str:
    """
    Firestore document ID for the questionnaire cache.
    Colons and slashes replaced — Firestore IDs cannot contain '/'.
    Format: pa-q--{payer_id}--{cpt_code}
    """
    safe_payer = payer_id.replace("/", "-").replace(":", "-")
    safe_cpt = cpt_code.replace("/", "-").replace(":", "-")
    return f"pa-q--{safe_payer}--{safe_cpt}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fetch_questionnaire(
    payer_id: str,
    cpt_code: str,
    config: Optional[CDSSConfig] = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Fetch the payer's DTR Questionnaire for a given CPT code.

    Returns a FHIR Questionnaire resource dict.
    The 'item' array in the returned dict is passed to PA-3.

    Args:
        payer_id:       Payer ID from CoverageCheckResult.payer_id
        cpt_code:       CPT/HCPCS code being authorized
        config:         CDSSConfig (defaults to singleton)
        force_refresh:  Bypass cache and fetch fresh from payer

    Returns:
        FHIR Questionnaire resource dict

    Raises:
        DTRFetchError: If questionnaire cannot be fetched or found
    """
    cfg = config or get_config()
    key = _cache_key(payer_id, cpt_code)

    # Check Firestore cache
    if not force_refresh:
        cached = await _get_from_cache(key, cfg)
        if cached:
            logger.info(
                "PA-2: Cache HIT payer=%s CPT=%s", payer_id, cpt_code
            )
            return cached

    logger.info(
        "PA-2: Cache MISS payer=%s CPT=%s — fetching from payer",
        payer_id, cpt_code,
    )

    # Fetch from payer
    questionnaire = await _fetch_from_payer(payer_id, cpt_code, cfg)

    # Store in cache (non-fatal if it fails)
    await _store_in_cache(key, questionnaire, cfg)

    return questionnaire


# ---------------------------------------------------------------------------
# Firestore cache
# ---------------------------------------------------------------------------

async def _get_from_cache(
    cache_key: str,
    config: CDSSConfig,
) -> Optional[dict[str, Any]]:
    """
    Retrieve a cached Questionnaire from Firestore.
    Returns None if not found or TTL expired.

    Firestore Python client is synchronous — we run it in an executor
    to avoid blocking the async event loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_get() -> Optional[dict[str, Any]]:
        try:
            db = firestore.Client(project=config.gcp_project_id)
            doc = db.collection(
                config.firestore_collection_questionnaires
            ).document(cache_key).get()

            if not doc.exists:
                return None

            data = doc.to_dict()
            cached_at = data.get("cached_at")

            if cached_at:
                ttl = timedelta(hours=config.dtr_cache_ttl_hours)
                age = datetime.now(timezone.utc) - cached_at
                if age > ttl:
                    logger.info(
                        "PA-2: Cache expired (age=%s TTL=%dh)",
                        age, config.dtr_cache_ttl_hours,
                    )
                    return None

            raw = data.get("questionnaire_json")
            return json.loads(raw) if raw else None

        except Exception as exc:
            logger.warning("PA-2: Cache read failed: %s", exc)
            return None

    return await loop.run_in_executor(None, _sync_get)


async def _store_in_cache(
    cache_key: str,
    questionnaire: dict[str, Any],
    config: CDSSConfig,
) -> None:
    """
    Store a Questionnaire in Firestore with a timestamp.
    Cache write failure is non-fatal — log and continue.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_set() -> None:
        try:
            db = firestore.Client(project=config.gcp_project_id)
            db.collection(
                config.firestore_collection_questionnaires
            ).document(cache_key).set({
                "questionnaire_json": json.dumps(questionnaire),
                "cached_at": datetime.now(timezone.utc),
                "questionnaire_id": questionnaire.get("id", "unknown"),
                "version": questionnaire.get("version", "unknown"),
            })
            logger.info("PA-2: Questionnaire cached (key=%s)", cache_key)
        except Exception as exc:
            logger.warning("PA-2: Cache write failed (non-fatal): %s", exc)

    await loop.run_in_executor(None, _sync_set)


# ---------------------------------------------------------------------------
# Payer DTR endpoint
# ---------------------------------------------------------------------------

async def _fetch_from_payer(
    payer_id: str,
    cpt_code: str,
    config: CDSSConfig,
) -> dict[str, Any]:
    """
    Fetch the FHIR Questionnaire from the payer's DTR endpoint.

    Tries the payer's FHIR endpoint first.
    Falls back to local templates in data/questionnaire_templates/ if:
      - No endpoint configured for this payer
      - Endpoint call fails
      - Endpoint returns empty bundle
    """
    crd_base = config.payer_endpoints.get(payer_id, "")
    if crd_base:
        # DTR endpoint is typically the same base domain + /dtr
        dtr_base = crd_base.replace("/cds-services", "").rstrip("/") + "/dtr"
        questionnaire = await _try_payer_dtr_endpoint(dtr_base, cpt_code, payer_id)
        if questionnaire:
            return questionnaire

    logger.warning(
        "PA-2: Payer DTR endpoint unavailable for payer=%s. "
        "Falling back to local template for CPT=%s.",
        payer_id, cpt_code,
    )
    return _load_local_template(payer_id, cpt_code)


async def _try_payer_dtr_endpoint(
    dtr_base: str,
    cpt_code: str,
    payer_id: str,
) -> Optional[dict[str, Any]]:
    """
    GET {dtr_base}/Questionnaire?context-of-use={cpt_code}
    Returns the Questionnaire dict or None if the call fails.
    """
    url = f"{dtr_base}/Questionnaire"
    params = {"context-of-use": cpt_code, "_sort": "-date", "_count": "1"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if not resp.ok:
                    logger.warning(
                        "PA-2: DTR endpoint %s → %s for payer=%s CPT=%s",
                        url, resp.status, payer_id, cpt_code,
                    )
                    return None

                bundle = await resp.json()
                entries = bundle.get("entry", [])
                if entries:
                    q = entries[0].get("resource")
                    logger.info(
                        "PA-2: Fetched Questionnaire %s from payer=%s",
                        q.get("id", "?"), payer_id,
                    )
                    return q

                logger.warning(
                    "PA-2: Empty bundle from DTR endpoint payer=%s CPT=%s",
                    payer_id, cpt_code,
                )
                return None

    except aiohttp.ClientError as exc:
        logger.warning("PA-2: DTR call failed payer=%s: %s", payer_id, exc)
        return None


def _load_local_template(payer_id: str, cpt_code: str) -> dict[str, Any]:
    """
    Load a questionnaire template from data/questionnaire_templates/.

    Lookup order:
      1. {payer_id}_{cpt_code}.json   — payer + service specific
      2. {cpt_code}.json              — generic for this CPT code
      3. generic_pa.json              — fully generic fallback

    Raises DTRFetchError if no template exists at all.
    """
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "data", "questionnaire_templates",
    )
    template_dir = os.path.normpath(template_dir)

    candidates = [
        os.path.join(template_dir, f"{payer_id}_{cpt_code}.json"),
        os.path.join(template_dir, f"{cpt_code}.json"),
        os.path.join(template_dir, "generic_pa.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                template = json.load(f)
            logger.info("PA-2: Using local template: %s", os.path.basename(path))
            return template

    raise DTRFetchError(
        payer_id=payer_id,
        cpt_code=cpt_code,
        message=(
            f"No questionnaire found for payer='{payer_id}' CPT='{cpt_code}'. "
            f"No payer DTR endpoint available and no local template in {template_dir}. "
            f"Add a template file or configure the payer endpoint in Secret Manager."
        ),
    )


# ---------------------------------------------------------------------------
# Cache invalidation utility
# ---------------------------------------------------------------------------

async def invalidate_cache(
    payer_id: str,
    cpt_code: str,
    config: Optional[CDSSConfig] = None,
) -> bool:
    """
    Manually invalidate a cached questionnaire.

    Use when a payer updates their questionnaire — e.g. after a
    payer notification or quarterly refresh cycle.

    Returns True if the cache entry was deleted, False if not found.
    """
    import asyncio
    cfg = config or get_config()
    key = _cache_key(payer_id, cpt_code)
    loop = asyncio.get_event_loop()

    def _sync_delete() -> bool:
        try:
            db = firestore.Client(project=cfg.gcp_project_id)
            doc_ref = db.collection(
                cfg.firestore_collection_questionnaires
            ).document(key)
            if doc_ref.get().exists:
                doc_ref.delete()
                logger.info(
                    "PA-2: Cache invalidated payer=%s CPT=%s", payer_id, cpt_code
                )
                return True
            return False
        except Exception as exc:
            logger.error("PA-2: Cache invalidation failed: %s", exc)
            return False

    return await loop.run_in_executor(None, _sync_delete)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DTRFetchError(Exception):
    """Raised when the DTR questionnaire cannot be retrieved or found."""

    def __init__(self, payer_id: str, cpt_code: str, message: str):
        self.payer_id = payer_id
        self.cpt_code = cpt_code
        super().__init__(message)