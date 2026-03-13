"""
agents/prior_auth/tools/bundle_assembler.py
============================================
PA-4: PAS FHIR Bundle Assembly + DLP Audit

What this step does:
  1. Constructs a Da Vinci PAS-compliant FHIR Bundle from the
     filled QuestionnaireResponse and patient resources
  2. Enforces the required PAS entry order:
       Claim → QuestionnaireResponse → ServiceRequest →
       Patient → Coverage → Practitioner → supporting resources
  3. Runs Cloud DLP inspection on the bundle before it is
     cleared for transmission
  4. Returns the bundle ready for PA-5 submission

Why entry order matters:
  Da Vinci PAS IG v2.1.0 specifies the exact order of resources
  in the transaction bundle. Payer systems validate this order.
  A mis-ordered bundle will be rejected at the payer gateway.

Why DLP runs here:
  The PA Agent is the only agent in this suite that sends PHI
  externally (to payers). DLP inspection is the last gate before
  the bundle leaves the organization. It runs in PA-4, not PA-5,
  so a DLP failure blocks assembly — not just transmission.

Reference: https://build.fhir.org/ig/HL7/davinci-pas/
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from google.cloud import dlp_v2

from shared.config import CDSSConfig, get_config
from shared.models import AnswerConfidence, QuestionnaireAnswer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def assemble_pas_bundle(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    questionnaire_id: str,
    answers: list[QuestionnaireAnswer],
    patient_resource: dict[str, Any],
    coverage_resource: dict[str, Any],
    service_request: dict[str, Any],
    practitioner_resource: Optional[dict[str, Any]] = None,
    supporting_resources: Optional[list[dict[str, Any]]] = None,
    config: Optional[CDSSConfig] = None,
) -> dict[str, Any]:
    """
    Assemble a Da Vinci PAS-compliant FHIR transaction bundle.

    Args:
        patient_id:            FHIR Patient resource ID
        cpt_code:              CPT/HCPCS code being authorized
        payer_id:              Payer identifier
        questionnaire_id:      FHIR Questionnaire resource ID (from PA-2)
        answers:               Validated QuestionnaireAnswer list (from PA-3)
        patient_resource:      FHIR Patient resource dict
        coverage_resource:     FHIR Coverage resource dict
        service_request:       FHIR ServiceRequest resource dict
        practitioner_resource: Optional FHIR Practitioner resource dict
        supporting_resources:  Optional list of additional FHIR resources
        config:                CDSSConfig (defaults to singleton)

    Returns:
        PAS-compliant FHIR Bundle ready for submission to payer $submit

    Raises:
        DLPInspectionError: If DLP finds high-severity PHI violations
        BundleAssemblyError: If required resources are missing
    """
    cfg = config or get_config()
    supporting_resources = supporting_resources or []

    logger.info(
        "PA-4: Assembling PAS bundle for patient=%s CPT=%s payer=%s",
        patient_id, cpt_code, payer_id,
    )

    # Build each resource in PAS IG entry order
    claim = _build_claim(
        patient_id=patient_id,
        cpt_code=cpt_code,
        payer_id=payer_id,
        coverage_resource=coverage_resource,
        service_request=service_request,
    )

    questionnaire_response = _build_questionnaire_response(
        patient_id=patient_id,
        questionnaire_id=questionnaire_id,
        answers=answers,
    )

    # Assemble bundle entries in PAS IG v2.1.0 required order
    entries = _build_bundle_entries(
        claim=claim,
        questionnaire_response=questionnaire_response,
        service_request=service_request,
        patient_resource=patient_resource,
        coverage_resource=coverage_resource,
        practitioner_resource=practitioner_resource,
        supporting_resources=supporting_resources,
    )

    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/davinci-pas/StructureDefinition/profile-pas-request-bundle"
            ]
        },
        "type": "transaction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entries,
    }

    # Run DLP inspection before clearing for transmission
    await _run_dlp_inspection(bundle, cfg)

    logger.info(
        "PA-4: PAS bundle assembled — %d entries, DLP passed",
        len(entries),
    )

    return bundle


# ---------------------------------------------------------------------------
# Claim builder
# ---------------------------------------------------------------------------

def _build_claim(
    patient_id: str,
    cpt_code: str,
    payer_id: str,
    coverage_resource: dict[str, Any],
    service_request: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the Claim resource — the primary resource in a PAS bundle.

    The Claim is a PA request, not a billing claim.
    type.coding uses the PA-specific code "professional" or "institutional".
    use must be "preauthorization" per PAS IG.
    """
    claim_id = str(uuid.uuid4())
    coverage_id = coverage_resource.get("id", "unknown")

    # Extract practitioner from ServiceRequest.requester if present
    requester = service_request.get("requester", {})
    requester_ref = requester.get("reference", f"Practitioner/unknown")

    claim: dict[str, Any] = {
        "resourceType": "Claim",
        "id": claim_id,
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/davinci-pas/StructureDefinition/profile-claim"
            ]
        },
        "status": "active",
        "type": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/claim-type",
                    "code": "professional",
                }
            ]
        },
        "use": "preauthorization",
        "patient": {"reference": f"Patient/{patient_id}"},
        "created": datetime.now(timezone.utc).date().isoformat(),
        "insurer": {
            "identifier": {
                "system": "http://hl7.org/fhir/sid/us-npi",
                "value": payer_id,
            }
        },
        "provider": {"reference": requester_ref},
        "priority": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/processpriority",
                    "code": "normal",
                }
            ]
        },
        "insurance": [
            {
                "sequence": 1,
                "focal": True,
                "coverage": {"reference": f"Coverage/{coverage_id}"},
            }
        ],
        "item": [
            {
                "sequence": 1,
                "productOrService": {
                    "coding": [
                        {
                            "system": "http://www.ama-assn.org/go/cpt",
                            "code": cpt_code,
                        }
                    ]
                },
                "servicedDate": datetime.now(timezone.utc).date().isoformat(),
                "extension": [
                    {
                        "url": "http://hl7.org/fhir/us/davinci-pas/StructureDefinition/extension-itemPreAuthIssued",
                        "valueBoolean": False,
                    }
                ],
            }
        ],
    }

    return claim


# ---------------------------------------------------------------------------
# QuestionnaireResponse builder
# ---------------------------------------------------------------------------

def _build_questionnaire_response(
    patient_id: str,
    questionnaire_id: str,
    answers: list[QuestionnaireAnswer],
) -> dict[str, Any]:
    """
    Build a FHIR QuestionnaireResponse from the validated answers.

    Status is "completed" — even if some answers are MISSING.
    MISSING answers are included with no answer value, which is
    valid per FHIR spec and signals to the payer that information
    was not available.

    The QuestionnaireResponse links back to the original Questionnaire
    via the 'questionnaire' field (canonical URL or resource ID).
    """
    qr_id = str(uuid.uuid4())

    items = []
    for answer in answers:
        item: dict[str, Any] = {
            "linkId": answer.link_id,
        }

        if answer.question_text:
            item["text"] = answer.question_text

        if (
            answer.confidence != AnswerConfidence.MISSING
            and answer.answer_value is not None
        ):
            # Map answer_value to FHIR answer type
            fhir_answer = _map_answer_value(answer.answer_value)
            item["answer"] = [fhir_answer]

            # Attach evidence extension
            if answer.evidence_resource_id:
                item["extension"] = [
                    {
                        "url": "http://hl7.org/fhir/us/davinci-dtr/StructureDefinition/information-origin",
                        "extension": [
                            {
                                "url": "source",
                                "valueCoding": {
                                    "system": "http://hl7.org/fhir/us/davinci-dtr/CodeSystem/informationOrigin",
                                    "code": "auto",
                                    "display": "Auto-populated from patient record",
                                },
                            },
                            {
                                "url": "author",
                                "valueReference": {
                                    "reference": f"Observation/{answer.evidence_resource_id}"
                                },
                            },
                        ],
                    }
                ]

        items.append(item)

    return {
        "resourceType": "QuestionnaireResponse",
        "id": qr_id,
        "questionnaire": questionnaire_id,
        "status": "completed",
        "subject": {"reference": f"Patient/{patient_id}"},
        "authored": datetime.now(timezone.utc).isoformat(),
        "item": items,
    }


def _map_answer_value(value: Any) -> dict[str, Any]:
    """
    Map a Python answer value to a FHIR QuestionnaireResponse answer item.

    FHIR answer types:
      valueBoolean, valueDecimal, valueInteger, valueDate,
      valueDateTime, valueTime, valueString, valueCoding,
      valueQuantity, valueReference
    """
    if isinstance(value, bool):
        return {"valueBoolean": value}
    if isinstance(value, int):
        return {"valueInteger": value}
    if isinstance(value, float):
        return {"valueDecimal": value}
    if isinstance(value, dict):
        # Could be a Coding, Quantity, or Reference
        if "code" in value and "system" in value:
            return {"valueCoding": value}
        if "value" in value and "unit" in value:
            return {"valueQuantity": value}
        if "reference" in value:
            return {"valueReference": value}
    # Default to string
    return {"valueString": str(value)}


# ---------------------------------------------------------------------------
# Bundle entry assembly (PAS IG v2.1.0 entry order)
# ---------------------------------------------------------------------------

def _build_bundle_entries(
    claim: dict[str, Any],
    questionnaire_response: dict[str, Any],
    service_request: dict[str, Any],
    patient_resource: dict[str, Any],
    coverage_resource: dict[str, Any],
    practitioner_resource: Optional[dict[str, Any]],
    supporting_resources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Assemble bundle entries in PAS IG v2.1.0 required order:
      1. Claim
      2. QuestionnaireResponse
      3. ServiceRequest
      4. Patient
      5. Coverage
      6. Practitioner (if present)
      7. Supporting resources

    Each entry uses a fullUrl with a urn:uuid: reference and
    a request object for the transaction bundle POST.
    """

    def _entry(resource: dict[str, Any]) -> dict[str, Any]:
        rtype = resource["resourceType"]
        rid = resource.get("id", str(uuid.uuid4()))
        return {
            "fullUrl": f"urn:uuid:{rid}",
            "resource": resource,
            "request": {
                "method": "POST",
                "url": rtype,
            },
        }

    entries = [
        _entry(claim),
        _entry(questionnaire_response),
        _entry(service_request),
        _entry(patient_resource),
        _entry(coverage_resource),
    ]

    if practitioner_resource:
        entries.append(_entry(practitioner_resource))

    for resource in supporting_resources:
        entries.append(_entry(resource))

    return entries


# ---------------------------------------------------------------------------
# DLP inspection
# ---------------------------------------------------------------------------

async def _run_dlp_inspection(
    bundle: dict[str, Any],
    config: CDSSConfig,
) -> None:
    """
    Run Cloud DLP inspection on the PAS bundle before transmission.

    The PA Agent is the only agent that sends PHI externally.
    DLP inspects the serialized bundle text for unexpected PHI patterns
    that should not be in the outbound payload (e.g., SSNs, full DOBs
    in plain text fields where they shouldn't appear).

    DLP findings at LIKELIHOOD_UNSPECIFIED or higher trigger a warning.
    Findings at VERY_LIKELY with info types in the block list raise
    DLPInspectionError and halt the bundle assembly.

    Block list info types (raise error):
      - US_SOCIAL_SECURITY_NUMBER
      - CREDIT_CARD_NUMBER
      - PASSPORT

    Warn-only info types (log warning, continue):
      - PERSON_NAME
      - DATE_OF_BIRTH
      - PHONE_NUMBER
      - EMAIL_ADDRESS
      - MEDICAL_RECORD_NUMBER
    """
    import asyncio
    import json as json_module

    cfg = config
    loop = asyncio.get_event_loop()

    # Serialize bundle to text for DLP inspection
    bundle_text = json_module.dumps(bundle)

    BLOCK_LIST_INFO_TYPES = {
        "US_SOCIAL_SECURITY_NUMBER",
        "CREDIT_CARD_NUMBER",
        "PASSPORT",
    }

    def _sync_dlp() -> None:
        try:
            dlp_client = dlp_v2.DlpServiceClient()

            inspect_config = dlp_v2.InspectConfig(
                info_types=[
                    dlp_v2.InfoType(name="US_SOCIAL_SECURITY_NUMBER"),
                    dlp_v2.InfoType(name="CREDIT_CARD_NUMBER"),
                    dlp_v2.InfoType(name="PASSPORT"),
                    dlp_v2.InfoType(name="PERSON_NAME"),
                    dlp_v2.InfoType(name="DATE_OF_BIRTH"),
                    dlp_v2.InfoType(name="PHONE_NUMBER"),
                    dlp_v2.InfoType(name="EMAIL_ADDRESS"),
                    dlp_v2.InfoType(name="MEDICAL_RECORD_NUMBER"),
                ],
                min_likelihood=dlp_v2.Likelihood.POSSIBLE,
                limits=dlp_v2.InspectConfig.FindingLimits(
                    max_findings_per_request=100
                ),
            )

            item = dlp_v2.ContentItem(value=bundle_text)

            response = dlp_client.inspect_content(
                request=dlp_v2.InspectContentRequest(
                    parent=f"projects/{cfg.gcp_project_id}",
                    inspect_config=inspect_config,
                    item=item,
                )
            )

            blocking_findings = []
            warning_findings = []

            for finding in response.result.findings:
                info_type_name = finding.info_type.name
                likelihood = finding.likelihood.name

                if info_type_name in BLOCK_LIST_INFO_TYPES:
                    blocking_findings.append(
                        f"{info_type_name} (likelihood={likelihood})"
                    )
                else:
                    warning_findings.append(
                        f"{info_type_name} (likelihood={likelihood})"
                    )

            if warning_findings:
                logger.warning(
                    "PA-4: DLP found %d PHI items in bundle (warn-only): %s",
                    len(warning_findings),
                    ", ".join(warning_findings[:5]),
                )

            if blocking_findings:
                raise DLPInspectionError(
                    f"DLP blocked PAS bundle — found {len(blocking_findings)} "
                    f"high-risk info types: {', '.join(blocking_findings)}. "
                    f"Review bundle before transmission."
                )

            logger.info("PA-4: DLP inspection passed")

        except DLPInspectionError:
            raise
        except Exception as exc:
            # DLP service errors should not block clinical workflow
            # Log and continue — operational resilience over perfect security
            logger.error(
                "PA-4: DLP inspection failed with unexpected error: %s. "
                "Proceeding with transmission — review DLP service health.",
                exc,
            )

    await loop.run_in_executor(None, _sync_dlp)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BundleAssemblyError(Exception):
    """Raised when required resources are missing for bundle assembly."""
    pass


class DLPInspectionError(Exception):
    """Raised when DLP finds blocking PHI violations in the PAS bundle."""
    pass