"""
scripts/mock_payer_server.py
==============================
Local aiohttp server that mimics a Da Vinci-compliant payer.
Handles CRD (CDS Hooks), DTR (Questionnaire), and PAS ($submit) endpoints.

Run in a separate terminal before tests:
    python scripts/mock_payer_server.py

Configure response scenarios via ENV vars before starting:
    $env:MOCK_PA_DECISION = "approved"   # approved | denied | pended | pending
    $env:MOCK_CRD_STATUS  = "required"   # required | not_required | unknown
    python scripts/mock_payer_server.py

Endpoints:
    POST /crd                    CDS Hooks order-sign hook
    GET  /dtr/Questionnaire      DTR questionnaire fetch
    POST /fhir/Claim/$submit     PAS bundle submission
    GET  /fhir/ClaimResponse     PAS decision polling
    GET  /health                 Health check
    POST /admin/set-scenario     Change scenario without restart
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("mock-payer")

# ── Mutable scenario state ────────────────────────────────────────────────────
# Change these via /admin/set-scenario or ENV vars

scenario = {
    "crd_status":   os.environ.get("MOCK_CRD_STATUS", "required"),   # required | not_required | unknown
    "pa_decision":  os.environ.get("MOCK_PA_DECISION", "approved"),   # approved | denied | pended | pending
    "response_delay_seconds": 0,
}

# Simulate async payer — store submissions and serve decisions after N polls
submissions: dict[str, dict] = {}
POLLS_BEFORE_DECISION = int(os.environ.get("MOCK_POLLS_BEFORE_DECISION", "1"))


# ── CDS Hooks: CRD ───────────────────────────────────────────────────────────

async def handle_crd(request: web.Request) -> web.Response:
    """
    POST /crd
    Receives a CDS Hooks order-sign hook payload.
    Returns cards indicating PA requirement status.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    hook_instance = body.get("hookInstance", str(uuid.uuid4()))
    logger.info("CRD request: hookInstance=%s scenario=%s", hook_instance, scenario["crd_status"])

    if scenario["response_delay_seconds"] > 0:
        await asyncio.sleep(scenario["response_delay_seconds"])

    crd_status = scenario["crd_status"]

    if crd_status == "not_required":
        cards = [{
            "uuid": str(uuid.uuid4()),
            "summary": "Prior authorization is NOT required for this service.",
            "detail": "This service does not require prior authorization under the patient's current coverage.",
            "indicator": "info",
            "source": {"label": "Mock Payer CRD", "url": "http://localhost:8080"},
        }]
    elif crd_status == "required":
        cards = [{
            "uuid": str(uuid.uuid4()),
            "summary": "Prior Authorization Required",
            "detail": (
                "Prior authorization is required for CPT 95251 under this patient's plan. "
                "Please complete the prior authorization questionnaire."
            ),
            "indicator": "warning",
            "source": {"label": "Mock Payer CRD", "url": "http://localhost:8080"},
            "links": [{
                "label": "Start Prior Authorization",
                "url": "http://localhost:8080/dtr/launch",
                "type": "smart",
            }],
        }]
    else:
        # UNKNOWN — return empty cards
        cards = []

    return web.json_response({"cards": cards, "systemActions": []})


# ── DTR: Questionnaire fetch ──────────────────────────────────────────────────

async def handle_dtr_questionnaire(request: web.Request) -> web.Response:
    """
    GET /dtr/Questionnaire?context-of-use={cpt_code}
    Returns the payer's documentation template for the given CPT code.
    """
    cpt_code = request.rel_url.query.get("context-of-use", "95251")
    logger.info("DTR questionnaire request: cpt_code=%s", cpt_code)

    questionnaire = {
        "resourceType": "Questionnaire",
        "id": f"mock-payer-q-{cpt_code}",
        "url": f"http://localhost:8080/fhir/Questionnaire/mock-payer-q-{cpt_code}",
        "version": "2026.1",
        "name": f"MockPayerPA_{cpt_code}",
        "title": f"Mock Payer Prior Authorization — CPT {cpt_code}",
        "status": "active",
        "subjectType": ["Patient"],
        "item": [
            {
                "linkId": "q1",
                "text": "Does the patient have a confirmed diagnosis of Type 1 or Type 2 Diabetes Mellitus?",
                "type": "boolean",
                "required": True,
            },
            {
                "linkId": "q2",
                "text": "What is the patient's most recent HbA1c value and date?",
                "type": "string",
                "required": True,
            },
            {
                "linkId": "q3",
                "text": "Is the patient currently on insulin therapy (basal, bolus, or pump)?",
                "type": "boolean",
                "required": True,
            },
            {
                "linkId": "q4",
                "text": "What insulin product and dose is the patient currently prescribed?",
                "type": "string",
                "required": True,
            },
            {
                "linkId": "q5",
                "text": "Does the patient have any contraindications to CGM device use?",
                "type": "boolean",
                "required": False,
            },
            {
                "linkId": "q6",
                "text": "What is the clinical justification for CGM at this time?",
                "type": "string",
                "required": True,
            },
        ],
    }

    return web.json_response(questionnaire)


# ── PAS: Bundle submission ────────────────────────────────────────────────────

async def handle_pas_submit(request: web.Request) -> web.Response:
    """
    POST /fhir/Claim/$submit
    Receives the PAS FHIR bundle.
    Returns an immediate ClaimResponse (queued) or synchronous decision
    depending on scenario configuration.
    """
    try:
        bundle = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    submission_id = f"mock-submission-{uuid.uuid4().hex[:8]}"
    received_at = datetime.now(timezone.utc).isoformat()
    is_expedited = request.headers.get("X-PAS-Expedited", "false").lower() == "true"

    logger.info(
        "PAS submission received: id=%s expedited=%s decision_scenario=%s",
        submission_id, is_expedited, scenario["pa_decision"]
    )

    # Count bundle entries for validation logging
    entry_count = len(bundle.get("entry", []))
    logger.info("  Bundle entries: %d", entry_count)

    # Store for polling
    submissions[submission_id] = {
        "received_at": received_at,
        "poll_count": 0,
        "decision": scenario["pa_decision"],
        "expedited": is_expedited,
        "bundle_entry_count": entry_count,
    }

    # Always return a PENDING ClaimResponse initially (async pattern)
    claim_response = _build_claim_response(
        submission_id=submission_id,
        decision="pending",
        received_at=received_at,
    )

    return web.json_response(claim_response, status=200)


async def handle_claim_response_poll(request: web.Request) -> web.Response:
    """
    GET /fhir/ClaimResponse?request={submission_id}
    Polled by pas_submit.py every 15 minutes.
    Returns PENDING for the first N polls, then the configured decision.
    """
    submission_id = request.rel_url.query.get("request", "")

    if submission_id not in submissions:
        return web.json_response(
            {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []},
            status=200
        )

    sub = submissions[submission_id]
    sub["poll_count"] += 1

    logger.info(
        "ClaimResponse poll: id=%s poll=%d decision=%s",
        submission_id, sub["poll_count"], sub["decision"]
    )

    # Return PENDING for first N polls, then real decision
    if sub["poll_count"] <= POLLS_BEFORE_DECISION:
        decision = "pending"
    else:
        decision = sub["decision"]

    claim_response = _build_claim_response(
        submission_id=submission_id,
        decision=decision,
        received_at=sub["received_at"],
    )

    return web.json_response({
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [{"resource": claim_response}]
    })


def _build_claim_response(
    submission_id: str,
    decision: str,
    received_at: str,
) -> dict:
    """Build a FHIR ClaimResponse for the given decision scenario."""
    now = datetime.now(timezone.utc).isoformat()

    # Map decision to FHIR ClaimResponse fields
    outcome_map = {
        "approved": ("complete", "approved"),
        "denied":   ("error",    "denied"),
        "pended":   ("partial",  "pended"),
        "pending":  ("queued",   "pending"),
    }
    fhir_outcome, _ = outcome_map.get(decision, ("queued", "pending"))

    cr: dict = {
        "resourceType": "ClaimResponse",
        "id": f"cr-{submission_id}",
        "status": "active",
        "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/claim-type",
                              "code": "professional"}]},
        "use": "preauthorization",
        "patient": {"reference": "Patient/test-patient-thornton-001"},
        "created": now,
        "insurer": {"display": "Mock Payer — BCBS CA"},
        "outcome": fhir_outcome,
        "disposition": _disposition_text(decision),
    }

    if decision == "approved":
        cr["preAuthRef"] = f"AUTH-MOCK-{uuid.uuid4().hex[:6].upper()}"
        cr["preAuthPeriod"] = {
            "start": now[:10],
            "end": "2026-09-10"
        }

    if decision == "denied":
        cr["error"] = [{
            "code": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/adjudication-error",
                                  "code": "A001"}],
                     "text": "Service not medically necessary per plan criteria"}
        }]

    if decision == "pended":
        cr["processNote"] = [{
            "number": 1,
            "type": {"coding": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/processpriority",
                                              "code": "normal"}]}]},
            "text": (
                "Additional documentation required: "
                "(1) Most recent ophthalmology report; "
                "(2) Documentation of previous glucose monitoring attempts; "
                "(3) Treating physician attestation of medical necessity."
            )
        }]

    return cr


def _disposition_text(decision: str) -> str:
    return {
        "approved": "Prior authorization approved. Auth number issued.",
        "denied": "Prior authorization denied. Service not medically necessary per plan criteria.",
        "pended": "Request pended. Additional documentation required. See processNote.",
        "pending": "Request received and queued for review.",
    }.get(decision, "Status unknown.")


# ── Admin: change scenario without restart ───────────────────────────────────

async def handle_set_scenario(request: web.Request) -> web.Response:
    """
    POST /admin/set-scenario
    Body: {"crd_status": "required", "pa_decision": "denied"}
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    if "crd_status" in body:
        scenario["crd_status"] = body["crd_status"]
    if "pa_decision" in body:
        scenario["pa_decision"] = body["pa_decision"]
    if "response_delay_seconds" in body:
        scenario["response_delay_seconds"] = body["response_delay_seconds"]

    logger.info("Scenario updated: %s", scenario)
    return web.json_response({"status": "ok", "scenario": scenario})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "scenario": scenario,
        "submissions_count": len(submissions),
        "polls_before_decision": POLLS_BEFORE_DECISION,
    })


# ── App setup ─────────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/crd", handle_crd)
    app.router.add_get("/dtr/Questionnaire", handle_dtr_questionnaire)
    app.router.add_post("/fhir/Claim/$submit", handle_pas_submit)
    app.router.add_get("/fhir/ClaimResponse", handle_claim_response_poll)
    app.router.add_post("/admin/set-scenario", handle_set_scenario)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  Mock Payer Server starting on http://localhost:8080")
    print(f"  CRD status  : {scenario['crd_status']}")
    print(f"  PA decision : {scenario['pa_decision']}")
    print(f"  Polls before decision: {POLLS_BEFORE_DECISION}")
    print()
    print("  Endpoints:")
    print("    POST /crd                  CDS Hooks CRD")
    print("    GET  /dtr/Questionnaire    DTR questionnaire")
    print("    POST /fhir/Claim/$submit   PAS submission")
    print("    GET  /fhir/ClaimResponse   Decision polling")
    print("    POST /admin/set-scenario   Change scenario live")
    print("    GET  /health               Health check")
    print()
    print("  Scenario shortcuts (run in separate terminal):")
    print('    Invoke-RestMethod -Uri http://localhost:8080/admin/set-scenario \\')
    print('      -Method POST -ContentType "application/json" \\')
    print('      -Body \'{"pa_decision": "denied"}\'')
    print("=" * 55)
    print()

    app = create_app()
    web.run_app(app, host="localhost", port=8080, access_log=None)
