"""
langgraph_prior_auth/graph.py

LangGraph StateGraph wrapping the existing async PA pipeline.
Adds deterministic state, checkpointing, and human-in-the-loop
interrupt before payer submission.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from agents.prior_auth.tools.coverage_check import check_coverage_requirements, PAStatus
from agents.prior_auth.tools.dtr_fetch import fetch_questionnaire
from agents.prior_auth.tools.bundle_assembler import assemble_pas_bundle
from agents.prior_auth.tools.questionnaire_filler import fill_questionnaire
from shared.fhir_client import FHIRClient
from shared.config import get_config

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class PAState(TypedDict):
    # Inputs
    patient_id: str
    cpt_code: str
    payer_id: str
    encounter_id: Optional[str]
    practitioner_id: Optional[str]
    # PA-1 outputs
    pa_required: Optional[bool]
    coverage_status: Optional[str]
    coverage_fhir_id: Optional[str]
    coverage_check_error: Optional[str]
    # PA-2 outputs
    questionnaire: Optional[dict]
    questionnaire_id: Optional[str]
    questionnaire_source: Optional[str]
    dtr_fetch_error: Optional[str]
    # PA-3 outputs
    answers: Optional[list]
    missing_required_count: Optional[int]
    filler_error: Optional[str]
    # PA-4 outputs
    pas_bundle: Optional[dict]
    pas_bundle_entries: Optional[int]
    dlp_blocked: Optional[bool]
    assembler_error: Optional[str]
    # PA-5 outputs
    claim_response_id: Optional[str]
    submit_error: Optional[str]
    # Final
    decision: Optional[Literal["APPROVED", "DENIED", "NOT_REQUIRED", "PENDING", "ERROR"]]
    decision_reason: Optional[str]
    # Internal
    _patient_bundle: Optional[dict]
    _clinical_impression: Optional[dict]
    _is_expedited: Optional[bool]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine in a dedicated thread with a fresh event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _get_cfg():
    """Return config with payer endpoints injected from env for local dev."""
    cfg = get_config()
    if not cfg.payer_endpoints:
        raw = os.environ.get("PAYER_ENDPOINTS", "{}")
        try:
            cfg.payer_endpoints = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("PAYER_ENDPOINTS env var is not valid JSON: %s", raw)
    return cfg


def _reconstruct_answers(raw_answers):
    """Reconstruct QuestionnaireAnswer objects from dicts after checkpoint serialization."""
    from shared.models import QuestionnaireAnswer, AnswerConfidence, EvidenceSource
    answers = []
    for a in (raw_answers or []):
        if isinstance(a, dict):
            # Reconstruct nested enums and dataclasses
            if "confidence" in a and isinstance(a["confidence"], str):
                try:
                    a = dict(a)
                    a["confidence"] = AnswerConfidence(a["confidence"])
                except ValueError:
                    pass
            if "evidence" in a and isinstance(a["evidence"], list):
                a = dict(a)
                reconstructed_evidence = []
                for e in a["evidence"]:
                    if isinstance(e, dict):
                        reconstructed_evidence.append(EvidenceSource(**e))
                    else:
                        reconstructed_evidence.append(e)
                a["evidence"] = reconstructed_evidence
            try:
                answers.append(QuestionnaireAnswer(**a))
            except Exception:
                answers.append(a)
        else:
            answers.append(a)
    return answers


def _extract_resource(bundle, resource_type, resource_id):
    if not bundle or not resource_id:
        return None
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        if r.get("resourceType") == resource_type and r.get("id") == resource_id:
            return r
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        if r.get("resourceType") == resource_type:
            return r
    return None


def _extract_coverage(bundle, coverage_id):
    return _extract_resource(bundle, "Coverage", coverage_id)


def _build_service_request(state):
    return {
        "resourceType": "ServiceRequest",
        "id": f"sr-{state['patient_id'][:8]}",
        "status": "active",
        "intent": "order",
        "code": {"coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": state["cpt_code"]}]},
        "subject": {"reference": f"Patient/{state['patient_id']}"},
        "encounter": {"reference": f"Encounter/{state['encounter_id']}"} if state.get("encounter_id") else None,
        "requester": {"reference": f"Practitioner/{state['practitioner_id']}"} if state.get("practitioner_id") else None,
    }


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_coverage_check(state: PAState) -> dict:
    """PA-1: Load FHIR bundle and check coverage requirements."""
    logger.info("PA-1 coverage_check | patient=%s cpt=%s", state["patient_id"], state["cpt_code"])
    cfg = _get_cfg()

    async def _coro():
        async with FHIRClient(cfg) as fhir_client:
            patient_bundle = await fhir_client.everything(
                patient_id=state["patient_id"],
                resource_types=[
                    "Condition", "Observation", "MedicationRequest",
                    "AllergyIntolerance", "Coverage", "DiagnosticReport",
                    "Encounter", "Procedure", "Practitioner",
                ],
            )
            impressions = await fhir_client.search(
                "ClinicalImpression",
                {"subject": f"Patient/{state['patient_id']}", "_sort": "-date", "_count": "1"},
            )
            clinical_impression = impressions[0] if impressions else {}
            result = await check_coverage_requirements(
                patient_id=state["patient_id"],
                cpt_code=state["cpt_code"],
                fhir_client=fhir_client,
                config=cfg,
                encounter_id=state.get("encounter_id"),
                practitioner_id=state.get("practitioner_id"),
            )
            return patient_bundle, clinical_impression, result

    try:
        patient_bundle, clinical_impression, coverage_result = _run(_coro())
        pa_required = coverage_result.status == PAStatus.REQUIRED
        update = {
            "pa_required": pa_required,
            "coverage_status": str(coverage_result.status.value),
            "coverage_fhir_id": getattr(coverage_result, "coverage_fhir_id", None),
            "coverage_check_error": None,
            "_patient_bundle": patient_bundle,
            "_clinical_impression": clinical_impression,
        }
        if not pa_required:
            update["decision"] = "NOT_REQUIRED"
            update["decision_reason"] = "Payer CRD indicates PA not required"
        return update
    except Exception as exc:
        logger.error("PA-1 failed: %s", exc, exc_info=True)
        return {
            "coverage_check_error": str(exc),
            "decision": "ERROR",
            "decision_reason": f"Coverage check failed: {exc}",
        }


def node_dtr_fetch(state: PAState) -> dict:
    """PA-2: Fetch DTR questionnaire from payer (Firestore cache)."""
    logger.info("PA-2 dtr_fetch | payer=%s cpt=%s", state["payer_id"], state["cpt_code"])
    cfg = _get_cfg()

    async def _coro():
        return await fetch_questionnaire(
            payer_id=state["payer_id"],
            cpt_code=state["cpt_code"],
            config=cfg,
        )

    try:
        questionnaire = _run(_coro())
        return {
            "questionnaire": questionnaire,
            "questionnaire_id": questionnaire.get("id", "local-template"),
            "questionnaire_source": "payer",
            "dtr_fetch_error": None,
        }
    except Exception as exc:
        logger.error("PA-2 failed: %s", exc, exc_info=True)
        return {
            "dtr_fetch_error": str(exc),
            "decision": "ERROR",
            "decision_reason": f"DTR fetch failed: {exc}",
        }


def node_questionnaire_filler(state: PAState) -> dict:
    """PA-3: Gemini fills questionnaire citing real FHIR resource IDs."""
    logger.info("PA-3 questionnaire_filler")
    cfg = _get_cfg()

    async def _coro():
        async with FHIRClient(cfg) as fhir_client:
            return await fill_questionnaire(
                questionnaire=state["questionnaire"],
                patient_bundle=state["_patient_bundle"],
                clinical_impression=state["_clinical_impression"],
                fhir_client=fhir_client,
                config=cfg,
            )

    try:
        answers = _run(_coro())
        missing = sum(
            1 for a in answers
            if getattr(a, "confidence", None) and str(a.confidence) in ("MISSING", "missing")
        )
        return {
            "answers": answers,
            "missing_required_count": missing,
            "filler_error": None,
        }
    except Exception as exc:
        logger.error("PA-3 failed: %s", exc, exc_info=True)
        return {
            "filler_error": str(exc),
            "decision": "ERROR",
            "decision_reason": f"Questionnaire filling failed: {exc}",
        }


def node_bundle_assembler(state: PAState) -> dict:
    """PA-4: Assemble Da Vinci PAS bundle and run Cloud DLP audit."""
    logger.info("PA-4 bundle_assembler + DLP")
    cfg = _get_cfg()

    async def _coro():
        # Reconstruct QuestionnaireAnswer objects in case checkpoint serialized them to dicts
        answers = _reconstruct_answers(state.get("answers", []))
        return await assemble_pas_bundle(
            patient_id=state["patient_id"],
            cpt_code=state["cpt_code"],
            payer_id=state["payer_id"],
            questionnaire_id=state.get("questionnaire_id", "unknown"),
            answers=answers,
            patient_resource=_extract_resource(
                state["_patient_bundle"], "Patient", state["patient_id"]
            ),
            coverage_resource=_extract_coverage(
                state["_patient_bundle"], state.get("coverage_fhir_id")
            ),
            service_request=_build_service_request(state),
            practitioner_resource=_extract_resource(
                state["_patient_bundle"], "Practitioner", state.get("practitioner_id")
            ),
            config=cfg,
        )

    try:
        pas_bundle = _run(_coro())
        return {
            "pas_bundle": pas_bundle,
            "pas_bundle_entries": len(pas_bundle.get("entry", [])),
            "dlp_blocked": False,
            "assembler_error": None,
        }
    except Exception as exc:
        logger.error("PA-4 failed: %s", exc, exc_info=True)
        dlp_blocked = "DLP" in str(exc) or "PHI" in str(exc)
        return {
            "dlp_blocked": dlp_blocked,
            "assembler_error": str(exc),
            "decision": "ERROR",
            "decision_reason": f"Bundle assembly failed: {exc}",
        }


def node_pas_submit(state: PAState) -> dict:
    """PA-5: Submit PAS bundle to payer, write ClaimResponse, publish Pub/Sub."""
    logger.info("PA-5 pas_submit | payer=%s", state["payer_id"])
    cfg = _get_cfg()

    from agents.prior_auth.tools.pas_submit import submit_pas_bundle

    async def _coro():
        async with FHIRClient(cfg) as fhir_client:
            return await submit_pas_bundle(
                pas_bundle=state["pas_bundle"],
                patient_id=state["patient_id"],
                cpt_code=state["cpt_code"],
                payer_id=state["payer_id"],
                fhir_client=fhir_client,
                config=cfg,
                is_expedited=state.get("_is_expedited", False),
            )

    try:
        decision = _run(_coro())
        raw = str(getattr(decision, "decision", "PENDING")).upper()
        final = raw if raw in ("APPROVED", "DENIED", "PENDING") else "PENDING"
        return {
            "claim_response_id": getattr(decision, "claim_response_id", None),
            "decision": final,
            "decision_reason": getattr(decision, "reason", ""),
            "submit_error": None,
        }
    except Exception as exc:
        logger.error("PA-5 failed: %s", exc, exc_info=True)
        return {
            "submit_error": str(exc),
            "decision": "ERROR",
            "decision_reason": f"PAS submission failed: {exc}",
        }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_coverage(state: PAState) -> str:
    if state.get("coverage_check_error") or state.get("pa_required") is False:
        return "__end__"
    return "dtr_fetch"


def route_after_questionnaire(state: PAState) -> str:
    if state.get("filler_error") or (state.get("missing_required_count") or 0) > 0:
        return "__end__"
    return "bundle_assembler"


def route_after_bundle(state: PAState) -> str:
    if state.get("assembler_error") or state.get("dlp_blocked"):
        return "__end__"
    return "pas_submit"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(PAState)

    builder.add_node("coverage_check", node_coverage_check)
    builder.add_node("dtr_fetch", node_dtr_fetch)
    builder.add_node("questionnaire_filler", node_questionnaire_filler)
    builder.add_node("bundle_assembler", node_bundle_assembler)
    builder.add_node("pas_submit", node_pas_submit)

    builder.set_entry_point("coverage_check")

    builder.add_conditional_edges(
        "coverage_check",
        route_after_coverage,
        {"dtr_fetch": "dtr_fetch", "__end__": END},
    )
    builder.add_edge("dtr_fetch", "questionnaire_filler")
    builder.add_conditional_edges(
        "questionnaire_filler",
        route_after_questionnaire,
        {"bundle_assembler": "bundle_assembler", "__end__": END},
    )
    builder.add_conditional_edges(
        "bundle_assembler",
        route_after_bundle,
        {"pas_submit": "pas_submit", "__end__": END},
    )
    builder.add_edge("pas_submit", END)

    return builder.compile(
        interrupt_before=["pas_submit"],
    )


graph = build_graph()
