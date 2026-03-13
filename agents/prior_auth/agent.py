"""
agents/prior_auth/agent.py
===========================
Prior Authorization Agent — Orchestrator

This is the main entry point for the PA Agent.
It wires together all five steps (PA-1 through PA-5) and manages
the human review gate before anything enters the permanent record.

Trigger:
  Subscribes to the 'orchestrator-ready' Pub/Sub topic.
  The upstream CDSS pipeline publishes a message here when a patient's
  FHIR record is ready with ClinicalImpression + CarePlan + DetectedIssue.

Flow:
  1. Receive Pub/Sub message → extract patient_id, cpt_code, payer_id
  2. Load patient FHIR bundle ($everything)
  3. PA-1: Check if PA is required (CRD)
     → NOT_REQUIRED: publish not-required event and exit
     → UNKNOWN: write Task for staff to verify coverage, exit
     → REQUIRED: continue to PA-2
  4. PA-2: Fetch payer questionnaire (DTR, with Firestore cache)
  5. PA-3: Fill questionnaire using Gemini + patient FHIR data
  6. Check for MISSING required answers
     → Any MISSING: write Task for clinician, write DocumentReference, exit
     → All answered: continue to PA-4
  7. PA-4: Assemble PAS bundle + DLP audit
  8. PA-5: Submit to payer + start async polling
  9. Write PAAgentResult to Firestore for audit log
  10. Publish result to prior-auth-ready Pub/Sub topic

Human review gate:
  ClaimResponse is written with status='draft' until a clinician
  reviews and promotes it to 'active'. The agent never directly
  triggers a care action — it only surfaces the PA decision.

Local development:
  Run directly with a synthetic Pub/Sub message using:
    python -m agents.prior_auth.agent --patient-id=test-123 \
        --cpt-code=99215 --payer-id=bcbs-ca-001

Cloud Run deployment:
  The agent runs as a Cloud Run service subscribed to the
  orchestrator-ready Pub/Sub push subscription.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from google.cloud import pubsub_v1, secretmanager

from agents.prior_auth.prompts import build_urgency_prompt
from agents.prior_auth.tools.bundle_assembler import (
    BundleAssemblyError,
    DLPInspectionError,
    assemble_pas_bundle,
)
from agents.prior_auth.tools.coverage_check import (
    CoverageCheckResult,
    check_coverage_requirements,
)
from agents.prior_auth.tools.dtr_fetch import DTRFetchError, fetch_questionnaire
from agents.prior_auth.tools.pas_submit import submit_pas_bundle
from agents.prior_auth.tools.questionnaire_filler import fill_questionnaire
from shared.config import CDSSConfig, get_config
from shared.fhir_client import FHIRClient
from shared.models import (
    AnswerConfidence,
    ClaimDecision,
    PAAgentResult,
    PAStatus,
    PATaskItem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

async def run_pa_agent(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    encounter_id: Optional[str] = None,
    practitioner_id: Optional[str] = None,
    care_plan: Optional[dict[str, Any]] = None,
    detected_issues: Optional[list[dict[str, Any]]] = None,
    config: Optional[CDSSConfig] = None,
) -> PAAgentResult:
    """
    Run the full Prior Authorization Agent pipeline for one patient + service.

    Args:
        patient_id:       FHIR Patient resource ID
        cpt_code:         CPT/HCPCS code for the proposed service
        payer_id:         Payer identifier (from Coverage resource)
        encounter_id:     Optional Encounter resource ID
        practitioner_id:  Optional Practitioner resource ID
        care_plan:        CarePlan FHIR resource (for urgency classification)
        detected_issues:  DetectedIssue resources (drug interactions/alerts)
        config:           CDSSConfig (defaults to singleton)

    Returns:
        PAAgentResult — complete audit record of the PA run
    """
    cfg = config or get_config()
    detected_issues = detected_issues or []

    result = PAAgentResult(
        patient_id=patient_id,
        encounter_id=encounter_id,
        cpt_code=cpt_code,
        payer_id=payer_id,
        initiated_at=datetime.now(timezone.utc),
    )

    logger.info(
        "PA Agent starting — patient=%s CPT=%s payer=%s",
        patient_id, cpt_code, payer_id,
    )

    async with FHIRClient(cfg) as fhir_client:

        # ── Load patient bundle ───────────────────────────────────────────────
        logger.info("Loading patient FHIR bundle")
        patient_bundle = await fhir_client.everything(
            patient_id=patient_id,
            resource_types=[
                "Condition", "Observation", "MedicationRequest",
                "AllergyIntolerance", "Coverage", "DiagnosticReport",
                "Encounter", "Procedure", "Practitioner",
            ],
        )

        # Extract ClinicalImpression (written by upstream pipeline)
        clinical_impressions = await fhir_client.search(
            "ClinicalImpression",
            {
                "subject": f"Patient/{patient_id}",
                "_sort": "-date",
                "_count": "1",
            },
        )
        clinical_impression = clinical_impressions[0] if clinical_impressions else {}

        if not clinical_impression:
            logger.warning(
                "PA Agent: No ClinicalImpression found for patient=%s. "
                "Proceeding without differential.",
                patient_id,
            )

        # ── PA-1: Coverage check ──────────────────────────────────────────────
        logger.info("PA-1: Checking coverage requirements")
        coverage_result: CoverageCheckResult = await check_coverage_requirements(
            patient_id=patient_id,
            cpt_code=cpt_code,
            fhir_client=fhir_client,
            config=cfg,
            encounter_id=encounter_id,
            practitioner_id=practitioner_id,
        )
        result.pa_required = coverage_result.status

        if coverage_result.status == PAStatus.NOT_REQUIRED:
            logger.info(
                "PA-1: PA NOT required for CPT=%s payer=%s — exiting",
                cpt_code, payer_id,
            )
            result.completed_at = datetime.now(timezone.utc)
            await _publish_not_required(patient_id, cpt_code, payer_id, cfg)
            return result

        if coverage_result.status == PAStatus.UNKNOWN:
            logger.warning(
                "PA-1: Coverage status UNKNOWN — writing Task for staff"
            )
            await _write_coverage_unknown_task(
                patient_id=patient_id,
                cpt_code=cpt_code,
                error_message=coverage_result.error_message,
                fhir_client=fhir_client,
            )
            result.completed_at = datetime.now(timezone.utc)
            return result

        # PA is REQUIRED — continue
        logger.info("PA-1: PA REQUIRED — proceeding to PA-2")

        # ── PA-2: Fetch questionnaire ─────────────────────────────────────────
        logger.info("PA-2: Fetching payer questionnaire")
        try:
            questionnaire = await fetch_questionnaire(
                payer_id=payer_id,
                cpt_code=cpt_code,
                config=cfg,
            )
            result.questionnaire_id = questionnaire.get("id", "local-template")
            logger.info(
                "PA-2: Questionnaire ready — ID=%s items=%d",
                result.questionnaire_id,
                len(questionnaire.get("item", [])),
            )
        except DTRFetchError as exc:
            logger.error("PA-2: DTR fetch failed: %s", exc)
            await _write_dtr_error_task(
                patient_id=patient_id,
                cpt_code=cpt_code,
                payer_id=payer_id,
                error=str(exc),
                fhir_client=fhir_client,
            )
            result.completed_at = datetime.now(timezone.utc)
            return result

        # ── PA-3: Fill questionnaire ──────────────────────────────────────────
        logger.info("PA-3: Filling questionnaire with Gemini")
        answers = await fill_questionnaire(
            questionnaire=questionnaire,
            patient_bundle=patient_bundle,
            clinical_impression=clinical_impression,
            fhir_client=fhir_client,
            config=cfg,
        )
        result.answers = answers

        missing_required = result.missing_required_count
        logger.info(
            "PA-3: %d answers generated, %d required MISSING",
            len(answers), missing_required,
        )

        # Check if missing required answers block submission
        if missing_required > 0:
            logger.warning(
                "PA-3: %d required questions unanswered — "
                "blocking submission, writing Task",
                missing_required,
            )
            missing_items = _extract_missing_items(answers)
            result.missing_items = missing_items
            result.blocked_by_missing = True

            task_id = await _write_missing_items_task(
                patient_id=patient_id,
                cpt_code=cpt_code,
                payer_id=payer_id,
                missing_items=missing_items,
                fhir_client=fhir_client,
            )
            result.task_fhir_id = task_id
            result.completed_at = datetime.now(timezone.utc)
            return result

        # ── Classify urgency ──────────────────────────────────────────────────
        is_expedited = await _classify_urgency(
            clinical_impression=clinical_impression,
            care_plan=care_plan or {},
            detected_issues=detected_issues,
            config=cfg,
        )
        logger.info(
            "Urgency classification: %s",
            "EXPEDITED" if is_expedited else "STANDARD",
        )

        # ── PA-4: Assemble PAS bundle ─────────────────────────────────────────
        logger.info("PA-4: Assembling PAS bundle")
        try:
            # Extract required resources from bundle
            patient_resource = _extract_patient(patient_bundle, patient_id)
            coverage_resource = _extract_coverage(
                patient_bundle, coverage_result.coverage_fhir_id
            )
            service_request = _build_service_request(
                patient_id=patient_id,
                cpt_code=cpt_code,
                encounter_id=encounter_id,
                practitioner_id=practitioner_id,
                coverage_id=coverage_result.coverage_fhir_id,
            )
            practitioner_resource = _extract_practitioner(
                patient_bundle, practitioner_id
            )

            pas_bundle = await assemble_pas_bundle(
                patient_id=patient_id,
                cpt_code=cpt_code,
                payer_id=payer_id,
                questionnaire_id=result.questionnaire_id or "unknown",
                answers=answers,
                patient_resource=patient_resource,
                coverage_resource=coverage_resource,
                service_request=service_request,
                practitioner_resource=practitioner_resource,
                config=cfg,
            )
            logger.info(
                "PA-4: PAS bundle assembled — %d entries",
                len(pas_bundle.get("entry", [])),
            )

        except DLPInspectionError as exc:
            logger.error("PA-4: DLP blocked bundle: %s", exc)
            await _write_dlp_error_task(
                patient_id=patient_id,
                cpt_code=cpt_code,
                error=str(exc),
                fhir_client=fhir_client,
            )
            result.completed_at = datetime.now(timezone.utc)
            return result

        except BundleAssemblyError as exc:
            logger.error("PA-4: Bundle assembly failed: %s", exc)
            result.completed_at = datetime.now(timezone.utc)
            return result

        # ── PA-5: Submit to payer ─────────────────────────────────────────────
        logger.info("PA-5: Submitting PAS bundle to payer")
        decision = await submit_pas_bundle(
            pas_bundle=pas_bundle,
            patient_id=patient_id,
            cpt_code=cpt_code,
            payer_id=payer_id,
            fhir_client=fhir_client,
            config=cfg,
            is_expedited=is_expedited,
        )
        result.decision = decision
        result.submission_id = decision.raw_claim_response.get("id") if decision.raw_claim_response else None

        logger.info(
            "PA Agent complete — patient=%s CPT=%s decision=%s",
            patient_id, cpt_code, decision.decision,
        )

    result.completed_at = datetime.now(timezone.utc)
    return result


# ---------------------------------------------------------------------------
# Pub/Sub subscription handler
# ---------------------------------------------------------------------------

def handle_pubsub_message(message: Any) -> None:
    """
    Handle an incoming Pub/Sub message from the orchestrator-ready topic.

    Message data format:
    {
        "patient_id":       "patient-123",
        "cpt_code":         "99215",
        "payer_id":         "bcbs-ca-001",
        "encounter_id":     "enc-456",      (optional)
        "practitioner_id":  "pract-789"     (optional)
    }

    This function is the Cloud Run push subscription handler.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))
        logger.info("Received Pub/Sub message: %s", data)

        patient_id = data.get("patient_id")
        cpt_code = data.get("cpt_code")
        payer_id = data.get("payer_id")

        if not all([patient_id, cpt_code, payer_id]):
            logger.error(
                "Invalid Pub/Sub message — missing required fields: %s", data
            )
            message.nack()
            return

        result = asyncio.run(
            run_pa_agent(
                patient_id=patient_id,
                cpt_code=cpt_code,
                payer_id=payer_id,
                encounter_id=data.get("encounter_id"),
                practitioner_id=data.get("practitioner_id"),
            )
        )

        logger.info(
            "PA Agent completed via Pub/Sub — patient=%s decision=%s",
            patient_id,
            result.decision.decision if result.decision else "no-decision",
        )
        message.ack()

    except Exception as exc:
        logger.error("PA Agent Pub/Sub handler failed: %s", exc, exc_info=True)
        message.nack()


# ---------------------------------------------------------------------------
# Urgency classifier
# ---------------------------------------------------------------------------

async def _classify_urgency(
    clinical_impression: dict[str, Any],
    care_plan: dict[str, Any],
    detected_issues: list[dict[str, Any]],
    config: CDSSConfig,
) -> bool:
    """
    Call Gemini to classify PA urgency (STANDARD vs EXPEDITED).
    Returns True if EXPEDITED, False if STANDARD.
    Falls back to STANDARD on error — conservative default.
    """
    import asyncio
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    import vertexai

    system_prompt, user_prompt = build_urgency_prompt(
        clinical_impression=clinical_impression,
        care_plan=care_plan,
        detected_issues=detected_issues,
    )

    loop = asyncio.get_event_loop()

    def _sync_call() -> bool:
        try:
            vertexai.init(
                project=config.gcp_project_id,
                location=config.gcp_region,
            )
            model = GenerativeModel(
                model_name=config.gemini_model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                contents=user_prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text.strip())
            urgency = parsed.get("urgency", "STANDARD").upper()
            logger.info(
                "Urgency: %s — %s", urgency, parsed.get("rationale", "")
            )
            return urgency == "EXPEDITED"
        except Exception as exc:
            logger.warning(
                "Urgency classification failed: %s — defaulting to STANDARD", exc
            )
            return False

    return await loop.run_in_executor(None, _sync_call)


# ---------------------------------------------------------------------------
# Resource extractors
# ---------------------------------------------------------------------------

def _extract_patient(
    bundle: dict[str, Any], patient_id: str
) -> dict[str, Any]:
    """Extract Patient resource from bundle or return minimal stub."""
    from shared.fhir_client import FHIRClient
    patients = FHIRClient.extract_resources(bundle, "Patient")
    for p in patients:
        if p.get("id") == patient_id:
            return p
    return {"resourceType": "Patient", "id": patient_id}


def _extract_coverage(
    bundle: dict[str, Any], coverage_id: Optional[str]
) -> dict[str, Any]:
    """Extract Coverage resource from bundle."""
    from shared.fhir_client import FHIRClient
    coverages = FHIRClient.extract_resources(bundle, "Coverage")
    if coverage_id:
        for c in coverages:
            if c.get("id") == coverage_id:
                return c
    if coverages:
        return coverages[0]
    return {"resourceType": "Coverage", "id": "unknown"}


def _extract_practitioner(
    bundle: dict[str, Any], practitioner_id: Optional[str]
) -> Optional[dict[str, Any]]:
    """Extract Practitioner resource from bundle if available."""
    if not practitioner_id:
        return None
    from shared.fhir_client import FHIRClient
    practitioners = FHIRClient.extract_resources(bundle, "Practitioner")
    for p in practitioners:
        if p.get("id") == practitioner_id:
            return p
    return None


def _build_service_request(
    patient_id: str,
    cpt_code: str,
    encounter_id: Optional[str],
    practitioner_id: Optional[str],
    coverage_id: Optional[str],
) -> dict[str, Any]:
    """Build a draft ServiceRequest for the proposed service."""
    sr: dict[str, Any] = {
        "resourceType": "ServiceRequest",
        "id": f"sr-{patient_id}-{cpt_code}",
        "status": "draft",
        "intent": "proposal",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [
                {
                    "system": "http://www.ama-assn.org/go/cpt",
                    "code": cpt_code,
                }
            ]
        },
    }
    if encounter_id:
        sr["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    if practitioner_id:
        sr["requester"] = {"reference": f"Practitioner/{practitioner_id}"}
    if coverage_id:
        sr["insurance"] = [{"reference": f"Coverage/{coverage_id}"}]
    return sr


# ---------------------------------------------------------------------------
# Task writers
# ---------------------------------------------------------------------------

def _extract_missing_items(
    answers: list,
) -> list[PATaskItem]:
    """Convert MISSING QuestionnaireAnswers to PATaskItem list."""
    return [
        PATaskItem(
            link_id=a.link_id,
            question_text=a.question_text or a.link_id,
            missing_info_needed=a.missing_info_needed or "Information not found in patient record",
        )
        for a in answers
        if a.is_required and a.confidence == AnswerConfidence.MISSING
    ]


async def _write_missing_items_task(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    missing_items: list[PATaskItem],
    fhir_client: FHIRClient,
) -> str:
    """Write a Task listing the required questions that could not be answered."""
    notes = [
        {
            "text": (
                f"Q: {item.question_text} | "
                f"Needed: {item.missing_info_needed}"
            ),
            "time": datetime.now(timezone.utc).isoformat(),
        }
        for item in missing_items
    ]

    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "routine",
        "code": {
            "text": (
                f"Prior Authorization Blocked — {len(missing_items)} "
                f"Required Questions Unanswered"
            )
        },
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"PA submission for CPT {cpt_code} to payer {payer_id} is blocked. "
            f"The following required questionnaire items could not be answered "
            f"from the patient record. Please provide the missing information "
            f"and rerun the PA agent."
        ),
        "note": notes,
    }

    written = await fhir_client.create("Task", task)
    return written.get("id", "unknown")


async def _write_coverage_unknown_task(
    patient_id: str,
    cpt_code: str,
    error_message: Optional[str],
    fhir_client: FHIRClient,
) -> None:
    """Write a Task when coverage status cannot be determined."""
    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "routine",
        "code": {"text": "Prior Authorization — Coverage Verification Required"},
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"Cannot determine PA requirements for CPT {cpt_code}. "
            f"{error_message or 'Coverage resource not found in FHIR store.'} "
            f"Please verify patient insurance and rerun the PA agent."
        ),
    }
    await fhir_client.create("Task", task)


async def _write_dtr_error_task(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    error: str,
    fhir_client: FHIRClient,
) -> None:
    """Write a Task when the payer questionnaire cannot be fetched."""
    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "routine",
        "code": {"text": "Prior Authorization — Questionnaire Unavailable"},
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"Could not retrieve PA questionnaire for CPT {cpt_code} "
            f"from payer {payer_id}. Error: {error}"
        ),
    }
    await fhir_client.create("Task", task)


async def _write_dlp_error_task(
    patient_id: str,
    cpt_code: str,
    error: str,
    fhir_client: FHIRClient,
) -> None:
    """Write a Task when DLP blocks the PAS bundle."""
    task = {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "priority": "urgent",
        "code": {"text": "Prior Authorization — DLP Inspection Failed"},
        "for": {"reference": f"Patient/{patient_id}"},
        "authoredOn": datetime.now(timezone.utc).isoformat(),
        "description": (
            f"PA bundle for CPT {cpt_code} was blocked by Cloud DLP. "
            f"Review the bundle for unexpected PHI before resubmitting. "
            f"Detail: {error}"
        ),
    }
    await fhir_client.create("Task", task)


# ---------------------------------------------------------------------------
# Pub/Sub publisher for not-required result
# ---------------------------------------------------------------------------

async def _publish_not_required(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    config: CDSSConfig,
) -> None:
    """Notify downstream systems that PA is not required for this service."""
    import asyncio
    loop = asyncio.get_event_loop()

    message = {
        "patient_id": patient_id,
        "cpt_code": cpt_code,
        "payer_id": payer_id,
        "decision": "PA_NOT_REQUIRED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    def _sync_publish() -> None:
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(
                config.gcp_project_id,
                config.pubsub_topic_prior_auth_ready,
            )
            publisher.publish(
                topic_path,
                data=json.dumps(message).encode("utf-8"),
            ).result(timeout=10)
        except Exception as exc:
            logger.warning("Failed to publish not-required event: %s", exc)

    await loop.run_in_executor(None, _sync_publish)


# ---------------------------------------------------------------------------
# CLI entry point for local development
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Prior Authorization Agent locally"
    )
    parser.add_argument("--patient-id", required=True, help="FHIR Patient resource ID")
    parser.add_argument("--cpt-code", required=True, help="CPT/HCPCS code")
    parser.add_argument("--payer-id", required=True, help="Payer identifier")
    parser.add_argument("--encounter-id", default=None, help="FHIR Encounter ID")
    parser.add_argument("--practitioner-id", default=None, help="FHIR Practitioner ID")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    result = asyncio.run(
        run_pa_agent(
            patient_id=args.patient_id,
            cpt_code=args.cpt_code,
            payer_id=args.payer_id,
            encounter_id=args.encounter_id,
            practitioner_id=args.practitioner_id,
        )
    )

    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))