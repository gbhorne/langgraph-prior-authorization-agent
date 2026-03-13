"""
agents/prior_auth/tools/pas_submit.py
=======================================
PA-5: PAS Bundle Submission + Async Polling

What this step does:
  1. POSTs the PAS bundle to the payer's $submit endpoint
  2. Never blocks — always uses async Pub/Sub pattern
  3. Polls for a decision via Cloud Scheduler (every 15 min)
  4. Writes ClaimResponse to FHIR store when decision arrives
  5. If pended → writes Task with payer's missing item list
  6. If approved → writes ClaimResponse + publishes to prior-auth-ready
  7. If denied → writes ClaimResponse + Task for clinician review

CMS-0057-F decision windows:
  Expedited: 72 hours  (3 days)
  Standard:  168 hours (7 days)

Why ClaimResponse is always async:
  Payers almost never return a synchronous decision.
  Blocking the agent on a payer response would hold Cloud Run
  resources for up to 7 days. Pub/Sub + Cloud Scheduler polling
  is the correct pattern.

Reference: https://build.fhir.org/ig/HL7/davinci-pas/
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from google.cloud import pubsub_v1

from shared.config import CDSSConfig, get_config
from shared.fhir_client import FHIRClient
from shared.models import ClaimDecision, ClaimResponseDecision, PATaskItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point — initial submission
# ---------------------------------------------------------------------------

async def submit_pas_bundle(
    pas_bundle: dict[str, Any],
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    fhir_client: FHIRClient,
    config: Optional[CDSSConfig] = None,
    is_expedited: bool = False,
) -> ClaimResponseDecision:
    """
    Submit the PAS bundle to the payer and handle the response.

    This method returns immediately after submission.
    If the payer returns a synchronous decision (rare), it is
    processed and written to FHIR.
    If the payer returns a pending/async response (common), the
    submission ID is stored in Firestore and Cloud Scheduler
    handles polling via poll_for_decision().

    Args:
        pas_bundle:    Assembled PAS FHIR Bundle from PA-4
        patient_id:    FHIR Patient resource ID
        cpt_code:      CPT/HCPCS code being authorized
        payer_id:      Payer identifier
        fhir_client:   Initialized FHIRClient
        config:        CDSSConfig (defaults to singleton)
        is_expedited:  True if this is an expedited (urgent) PA request

    Returns:
        ClaimResponseDecision with initial status.
        decision=PENDING means the payer response is async —
        poll_for_decision() will update this via Cloud Scheduler.
    """
    cfg = config or get_config()
    submit_url = _get_submit_url(payer_id, cfg)

    if not submit_url:
        logger.error(
            "PA-5: No $submit endpoint configured for payer=%s", payer_id
        )
        return ClaimResponseDecision(
            decision=ClaimDecision.ERROR,
            denial_reason=(
                f"No $submit endpoint configured for payer '{payer_id}'. "
                f"Register this payer using scripts/setup_payer_secrets.py"
            ),
        )

    logger.info(
        "PA-5: Submitting PAS bundle to payer=%s endpoint=%s "
        "expedited=%s",
        payer_id, submit_url, is_expedited,
    )

    # POST to payer $submit endpoint
    raw_response, http_status = await _post_to_payer(
        submit_url=submit_url,
        bundle=pas_bundle,
        is_expedited=is_expedited,
    )

    if raw_response is None:
        return ClaimResponseDecision(
            decision=ClaimDecision.ERROR,
            denial_reason=f"HTTP {http_status} from payer $submit endpoint",
        )

    # Parse payer response
    decision = _parse_claim_response(raw_response)

    # Write ClaimResponse to FHIR store
    if decision.decision != ClaimDecision.PENDING:
        written = await _write_claim_response_to_fhir(
            claim_response=raw_response,
            patient_id=patient_id,
            fhir_client=fhir_client,
        )
        decision.claim_response_id = written.get("id")

    # Store submission state in Firestore for polling
    submission_id = raw_response.get("id") or str(uuid.uuid4())
    decision.raw_claim_response = raw_response

    await _store_submission_state(
        submission_id=submission_id,
        patient_id=patient_id,
        cpt_code=cpt_code,
        payer_id=payer_id,
        decision=decision,
        is_expedited=is_expedited,
        config=cfg,
    )

    # Handle pended — write Task with missing items
    if decision.decision == ClaimDecision.PENDED and decision.pended_items:
        task_id = await _write_pended_task(
            patient_id=patient_id,
            pended_items=decision.pended_items,
            submission_id=submission_id,
            fhir_client=fhir_client,
        )
        logger.info(
            "PA-5: Pended — wrote Task %s with %d missing items",
            task_id, len(decision.pended_items),
        )

    # Publish to Pub/Sub if we have a final decision already
    if decision.decision in (ClaimDecision.APPROVED, ClaimDecision.DENIED):
        await _publish_decision(
            patient_id=patient_id,
            cpt_code=cpt_code,
            payer_id=payer_id,
            decision=decision,
            config=cfg,
        )

    logger.info(
        "PA-5: Submission complete — decision=%s submission_id=%s",
        decision.decision, submission_id,
    )

    return decision


# ---------------------------------------------------------------------------
# Polling (called by Cloud Scheduler every 15 minutes)
# ---------------------------------------------------------------------------

async def poll_for_decision(
    submission_id: str,
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    fhir_client: FHIRClient,
    config: Optional[CDSSConfig] = None,
) -> ClaimResponseDecision:
    """
    Poll the payer for a decision on a pending PA submission.

    Called by Cloud Scheduler every pas_poll_interval_minutes (default 15).
    Checks whether the payer has issued a decision since last poll.

    If decision received:
      - Write ClaimResponse to FHIR
      - Publish to prior-auth-ready Pub/Sub topic
      - Update Firestore submission state

    If still pending:
      - Update poll count in Firestore
      - Check if we have exceeded the CMS decision window
      - If window exceeded → write escalation Task

    Args:
        submission_id:  Payer-assigned or internal submission tracking ID
        patient_id:     FHIR Patient resource ID
        cpt_code:       CPT/HCPCS code being authorized
        payer_id:       Payer identifier
        fhir_client:    Initialized FHIRClient
        config:         CDSSConfig (defaults to singleton)

    Returns:
        Updated ClaimResponseDecision
    """
    cfg = config or get_config()

    # Load submission state from Firestore
    state = await _load_submission_state(submission_id, cfg)
    if not state:
        logger.error(
            "PA-5: No submission state found for submission_id=%s",
            submission_id,
        )
        return ClaimResponseDecision(
            decision=ClaimDecision.ERROR,
            denial_reason=f"Submission state not found for ID {submission_id}",
        )

    poll_count = state.get("poll_count", 0) + 1
    is_expedited = state.get("is_expedited", False)
    submitted_at = state.get("submitted_at")

    # Check CMS decision window
    max_hours = (
        cfg.pas_max_poll_hours_expedited
        if is_expedited
        else cfg.pas_max_poll_hours_standard
    )

    if submitted_at:
        elapsed_hours = (
            datetime.now(timezone.utc) - submitted_at
        ).total_seconds() / 3600

        if elapsed_hours > max_hours:
            logger.warning(
                "PA-5: CMS decision window exceeded — "
                "submission_id=%s elapsed=%.1fh max=%dh",
                submission_id, elapsed_hours, max_hours,
            )
            await _write_window_exceeded_task(
                patient_id=patient_id,
                submission_id=submission_id,
                payer_id=payer_id,
                elapsed_hours=elapsed_hours,
                max_hours=max_hours,
                fhir_client=fhir_client,
            )

    # Query payer for decision
    submit_url = _get_submit_url(payer_id, cfg)
    raw_response = await _query_payer_status(
        submit_url=submit_url,
        submission_id=submission_id,
        payer_id=payer_id,
    )

    if raw_response is None:
        # Still pending or query failed — update poll count
        await _update_poll_count(submission_id, poll_count, cfg)
        return ClaimResponseDecision(
            decision=ClaimDecision.PENDING,
            poll_count=poll_count,
        )

    decision = _parse_claim_response(raw_response)
    decision.poll_count = poll_count
    decision.raw_claim_response = raw_response

    # Write ClaimResponse to FHIR
    written = await _write_claim_response_to_fhir(
        claim_response=raw_response,
        patient_id=patient_id,
        fhir_client=fhir_client,
    )
    decision.claim_response_id = written.get("id")

    # Update Firestore state
    await _update_submission_state(submission_id, decision, poll_count, cfg)

    # Publish final decision
    if decision.decision in (ClaimDecision.APPROVED, ClaimDecision.DENIED):
        await _publish_decision(
            patient_id=patient_id,
            cpt_code=cpt_code,
            payer_id=payer_id,
            decision=decision,
            config=cfg,
        )

    logger.info(
        "PA-5: Poll complete — submission_id=%s decision=%s poll_count=%d",
        submission_id, decision.decision, poll_count,
    )

    return decision


# ---------------------------------------------------------------------------
# HTTP submission
# ---------------------------------------------------------------------------

def _get_submit_url(payer_id: str, config: CDSSConfig) -> Optional[str]:
    """
    Look up the payer's $submit URL.
    PAS $submit is the CRD base URL + /fhir/Claim/$submit
    """
    crd_base = config.payer_endpoints.get(payer_id, "")
    if not crd_base:
        return None
    base = crd_base.replace("/cds-services", "").replace("/crd", "").rstrip("/")
    return f"{base}/fhir/Claim/$submit"


async def _post_to_payer(
    submit_url: str,
    bundle: dict[str, Any],
    is_expedited: bool,
) -> tuple[Optional[dict[str, Any]], int]:
    """
    POST the PAS bundle to the payer's $submit endpoint.

    Returns (response_dict, http_status).
    Returns (None, http_status) on failure.
    """
    headers = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }

    if is_expedited:
        # Signal expedited PA to the payer per PAS IG
        headers["X-PAS-Expedited"] = "true"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                submit_url,
                json=bundle,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                raw = await resp.text()
                if not resp.ok:
                    logger.error(
                        "PA-5: Payer $submit returned %s: %s",
                        resp.status, raw[:500],
                    )
                    return None, resp.status

                return json.loads(raw), resp.status

    except aiohttp.ClientError as exc:
        logger.error("PA-5: $submit HTTP error: %s", exc)
        return None, 0


async def _query_payer_status(
    submit_url: Optional[str],
    submission_id: str,
    payer_id: str,
) -> Optional[dict[str, Any]]:
    """
    Query the payer for the status of a pending submission.

    PAS IG defines a polling pattern via ClaimResponse resource read.
    GET {base}/fhir/ClaimResponse?request.identifier={submission_id}
    """
    if not submit_url:
        return None

    # Build status URL from submit URL
    status_url = submit_url.replace("Claim/$submit", "ClaimResponse")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                status_url,
                params={"request.identifier": submission_id},
                headers={"Accept": "application/fhir+json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if not resp.ok:
                    logger.debug(
                        "PA-5: Status query returned %s for submission_id=%s",
                        resp.status, submission_id,
                    )
                    return None

                bundle = await resp.json()
                entries = bundle.get("entry", [])
                if entries:
                    return entries[0].get("resource")
                return None

    except aiohttp.ClientError as exc:
        logger.warning("PA-5: Status query failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ClaimResponse parser
# ---------------------------------------------------------------------------

def _parse_claim_response(
    claim_response: dict[str, Any],
) -> ClaimResponseDecision:
    """
    Parse a FHIR ClaimResponse resource into a ClaimResponseDecision.

    ClaimResponse.outcome maps to ClaimDecision:
      complete   → APPROVED
      error      → DENIED
      partial    → PENDED (payer needs more info)
      queued     → PENDING (not yet processed)
    """
    outcome = claim_response.get("outcome", "queued").lower()

    outcome_map = {
        "complete": ClaimDecision.APPROVED,
        "error": ClaimDecision.DENIED,
        "partial": ClaimDecision.PENDED,
        "queued": ClaimDecision.PENDING,
    }
    decision = outcome_map.get(outcome, ClaimDecision.PENDING)

    auth_number = None
    denial_reason = None
    pended_items: list[str] = []

    # Extract auth number from preAuthRef
    for insurance in claim_response.get("insurance", []):
        pre_auth = insurance.get("preAuthRef", [])
        if pre_auth:
            auth_number = pre_auth[0]
            break

    # Extract denial reason from errors
    errors = claim_response.get("error", [])
    if errors:
        denial_reason = "; ".join(
            e.get("code", {}).get("text", "Unknown error")
            for e in errors[:3]
        )

    # Extract pended items from processNote
    for note in claim_response.get("processNote", []):
        text = note.get("text", "")
        if text:
            pended_items.append(text)

    # Also check items for adjudication notes
    for item in claim_response.get("item", []):
        for adj in item.get("adjudication", []):
            reason = adj.get("reason", {}).get("text", "")
            if reason and reason not in pended_items:
                pended_items.append(reason)

    return ClaimResponseDecision(
        decision=decision,
        auth_number=auth_number,
        denial_reason=denial_reason,
        pended_items=pended_items,
        decision_timestamp=(
            datetime.now(timezone.utc)
            if decision not in (ClaimDecision.PENDING, ClaimDecision.PENDED)
            else None
        ),
    )


# ---------------------------------------------------------------------------
# FHIR writers
# ---------------------------------------------------------------------------

async def _write_claim_response_to_fhir(
    claim_response: dict[str, Any],
    patient_id: str,
    fhir_client: FHIRClient,
) -> dict[str, Any]:
    """
    Write the ClaimResponse to the FHIR store.
    Ensures patient reference is set for record linkage.
    """
    if "patient" not in claim_response:
        claim_response["patient"] = {"reference": f"Patient/{patient_id}"}

    written = await fhir_client.create("ClaimResponse", claim_response)
    logger.info(
        "PA-5: ClaimResponse written to FHIR — ID=%s", written.get("id")
    )
    return written


async def _write_pended_task(
    patient_id: str,
    pended_items: list[str],
    submission_id: str,
    fhir_client: FHIRClient,
) -> str:
    """
    Write a FHIR Task when the payer pends the request.

    The Task lists the specific items the payer needs.
    Clinician or CDI team resolves the Task and resubmits.
    """
    notes = [
        {"text": item, "time": datetime.now(timezone.utc).isoformat()}
        for item in pended_items
    ]

    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "routine",
        "code": {
            "coding": [
                {
                    "system": "http://hl7.org/fhir/CodeSystem/task-code",
                    "code": "approve",
                    "display": "Activate/approve the focal resource",
                }
            ],
            "text": "Prior Authorization — Payer Pended: Additional Information Required",
        },
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "note": notes,
        "description": (
            f"Payer has pended PA submission {submission_id}. "
            f"Please provide the {len(pended_items)} item(s) listed in notes "
            f"and resubmit."
        ),
    }

    written = await fhir_client.create("Task", task)
    return written.get("id", "unknown")


async def _write_window_exceeded_task(
    patient_id: str,
    submission_id: str,
    payer_id: str,
    elapsed_hours: float,
    max_hours: int,
    fhir_client: FHIRClient,
) -> None:
    """
    Write an escalation Task when the CMS decision window is exceeded.
    This is a regulatory compliance flag — the payer is overdue.
    """
    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "urgent",
        "code": {
            "text": "Prior Authorization — CMS Decision Window Exceeded"
        },
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"Payer '{payer_id}' has not responded to PA submission "
            f"{submission_id} within the CMS-required {max_hours}-hour window. "
            f"Elapsed time: {elapsed_hours:.1f} hours. "
            f"Escalate to payer relations or submit a complaint per CMS-0057-F."
        ),
    }
    await fhir_client.create("Task", task)
    logger.warning(
        "PA-5: CMS window exceeded Task written for submission_id=%s",
        submission_id,
    )


# ---------------------------------------------------------------------------
# Pub/Sub publisher
# ---------------------------------------------------------------------------

async def _publish_decision(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    decision: ClaimResponseDecision,
    config: CDSSConfig,
) -> None:
    """
    Publish the final PA decision to the prior-auth-ready Pub/Sub topic.

    Downstream consumers (scheduling, care coordination, CDI) subscribe
    to this topic to act on the PA outcome.
    """
    import asyncio

    loop = asyncio.get_event_loop()

    message = {
        "patient_id": patient_id,
        "cpt_code": cpt_code,
        "payer_id": payer_id,
        "decision": decision.decision.value,
        "auth_number": decision.auth_number,
        "denial_reason": decision.denial_reason,
        "claim_response_id": decision.claim_response_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    def _sync_publish() -> None:
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(
                config.gcp_project_id,
                config.pubsub_topic_prior_auth_ready,
            )
            data = json.dumps(message).encode("utf-8")
            future = publisher.publish(topic_path, data=data)
            message_id = future.result(timeout=10)
            logger.info(
                "PA-5: Published decision to Pub/Sub — message_id=%s", message_id
            )
        except Exception as exc:
            logger.error("PA-5: Pub/Sub publish failed: %s", exc)

    await loop.run_in_executor(None, _sync_publish)


# ---------------------------------------------------------------------------
# Firestore state management
# ---------------------------------------------------------------------------

async def _store_submission_state(
    submission_id: str,
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    decision: ClaimResponseDecision,
    is_expedited: bool,
    config: CDSSConfig,
) -> None:
    """Store PA submission state in Firestore for polling continuity."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_set() -> None:
        try:
            from google.cloud import firestore
            db = firestore.Client(project=config.gcp_project_id)
            db.collection(config.firestore_collection_pa_status).document(
                submission_id
            ).set({
                "patient_id": patient_id,
                "cpt_code": cpt_code,
                "payer_id": payer_id,
                "decision": decision.decision.value,
                "is_expedited": is_expedited,
                "submitted_at": datetime.now(timezone.utc),
                "poll_count": 0,
                "claim_response_id": decision.claim_response_id,
            })
        except Exception as exc:
            logger.error("PA-5: Failed to store submission state: %s", exc)

    await loop.run_in_executor(None, _sync_set)


async def _load_submission_state(
    submission_id: str,
    config: CDSSConfig,
) -> Optional[dict[str, Any]]:
    """Load PA submission state from Firestore."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_get() -> Optional[dict[str, Any]]:
        try:
            from google.cloud import firestore
            db = firestore.Client(project=config.gcp_project_id)
            doc = db.collection(
                config.firestore_collection_pa_status
            ).document(submission_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as exc:
            logger.error("PA-5: Failed to load submission state: %s", exc)
            return None

    return await loop.run_in_executor(None, _sync_get)


async def _update_poll_count(
    submission_id: str,
    poll_count: int,
    config: CDSSConfig,
) -> None:
    """Increment poll count in Firestore."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_update() -> None:
        try:
            from google.cloud import firestore
            db = firestore.Client(project=config.gcp_project_id)
            db.collection(
                config.firestore_collection_pa_status
            ).document(submission_id).update({"poll_count": poll_count})
        except Exception as exc:
            logger.warning("PA-5: Failed to update poll count: %s", exc)

    await loop.run_in_executor(None, _sync_update)


async def _update_submission_state(
    submission_id: str,
    decision: ClaimResponseDecision,
    poll_count: int,
    config: CDSSConfig,
) -> None:
    """Update submission state with final decision."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _sync_update() -> None:
        try:
            from google.cloud import firestore
            db = firestore.Client(project=config.gcp_project_id)
            db.collection(
                config.firestore_collection_pa_status
            ).document(submission_id).update({
                "decision": decision.decision.value,
                "auth_number": decision.auth_number,
                "denial_reason": decision.denial_reason,
                "claim_response_id": decision.claim_response_id,
                "poll_count": poll_count,
                "decided_at": datetime.now(timezone.utc),
            })
        except Exception as exc:
            logger.warning("PA-5: Failed to update submission state: %s", exc)

    await loop.run_in_executor(None, _sync_update)