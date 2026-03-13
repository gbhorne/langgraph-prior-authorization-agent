"""
agents/prior_auth/tools/coverage_check.py
==========================================
PA-1: Coverage Requirements Discovery (CRD)

What this step does:
  1. Reads the patient's Coverage resource from the FHIR store
  2. Calls the payer's CRD endpoint (a CDS Hooks service) with the
     proposed CPT/HCPCS code from the CarePlan
  3. Parses the response to determine if PA is required
  4. Falls back to Availity eligibility API if Coverage is absent

Da Vinci CRD flow:
  - CRD is a CDS Hooks service the payer exposes
  - We fire an order-sign hook with the proposed service
  - Payer returns "cards" — one may contain a PA-required flag
  - PA required → proceed to PA-2 (DTR fetch)
  - PA not required → short-circuit, return NOT_REQUIRED

CMS-0057-F context:
  - CRD endpoints required for payers >10k members by Jan 1, 2026
  - Not all payers have compliant endpoints yet
  - Availity bridges the gap for non-FHIR payers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from shared.config import CDSSConfig, get_config
from shared.fhir_client import FHIRClient
from shared.models import PAStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoverageCheckResult:
    """
    Output of the PA-1 coverage check step.

    status:           PA_REQUIRED | PA_NOT_REQUIRED | UNKNOWN
    payer_id:         Extracted from Coverage.payor[0]
    coverage_fhir_id: Coverage resource ID (None if Availity fallback used)
    auth_number_hint: Pre-auth number if already on file (rare)
    raw_crd_response: Full CDS Hooks response for audit logging
    used_fallback:    True if Availity fallback was used
    error_message:    Set if UNKNOWN status due to an error
    """
    status: PAStatus
    payer_id: str
    coverage_fhir_id: Optional[str] = None
    auth_number_hint: Optional[str] = None
    raw_crd_response: Optional[dict[str, Any]] = None
    used_fallback: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def check_coverage_requirements(
    patient_id: str,
    cpt_code: str,
    fhir_client: FHIRClient,
    config: Optional[CDSSConfig] = None,
    encounter_id: Optional[str] = None,
    practitioner_id: Optional[str] = None,
) -> CoverageCheckResult:
    """
    Determine if a CPT/HCPCS code requires prior authorization.

    Flow:
      1. Search FHIR store for active Coverage resource
      2. If found  → call payer CRD endpoint
      3. If absent → call Availity eligibility API as fallback

    Args:
        patient_id:       FHIR Patient resource ID
        cpt_code:         CPT or HCPCS code for the proposed service
        fhir_client:      Initialized FHIRClient instance
        config:           CDSSConfig (defaults to singleton)
        encounter_id:     Optional Encounter ID (improves CRD accuracy)
        practitioner_id:  Optional Practitioner ID (ordering provider)

    Returns:
        CoverageCheckResult with PA status and payer identity
    """
    cfg = config or get_config()

    # Step 1 — read Coverage from FHIR
    coverage = await _read_coverage(patient_id, fhir_client)

    if coverage:
        payer_id = _extract_payer_id(coverage)
        logger.info(
            "PA-1: Coverage found for patient=%s payer=%s", patient_id, payer_id
        )

        # Step 2 — call CRD endpoint
        crd_result = await _call_crd_endpoint(
            payer_id=payer_id,
            patient_id=patient_id,
            cpt_code=cpt_code,
            coverage=coverage,
            config=cfg,
            encounter_id=encounter_id,
            practitioner_id=practitioner_id,
        )

        return CoverageCheckResult(
            status=crd_result["status"],
            payer_id=payer_id,
            coverage_fhir_id=coverage.get("id"),
            auth_number_hint=crd_result.get("auth_number_hint"),
            raw_crd_response=crd_result.get("raw_response"),
            used_fallback=False,
        )

    else:
        # Step 3 — Availity fallback
        logger.warning(
            "PA-1: No Coverage resource for patient=%s. "
            "Activating Availity fallback.",
            patient_id,
        )
        return await _availity_fallback(patient_id, cpt_code, cfg)


# ---------------------------------------------------------------------------
# Coverage resource reader
# ---------------------------------------------------------------------------

async def _read_coverage(
    patient_id: str,
    fhir_client: FHIRClient,
) -> Optional[dict[str, Any]]:
    """
    Search for an active Coverage resource for this patient.
    Returns the first active Coverage, or None if absent.

    A patient may have multiple Coverage resources (primary, secondary).
    We use the first active one. In production you may want to select
    based on the service category in the CarePlan.
    """
    try:
        coverages = await fhir_client.search(
            "Coverage",
            {
                "beneficiary": f"Patient/{patient_id}",
                "status": "active",
            },
        )
        if coverages:
            logger.debug(
                "PA-1: Found %d Coverage resource(s) for patient=%s",
                len(coverages), patient_id,
            )
            return coverages[0]
        return None

    except Exception as exc:
        logger.warning(
            "PA-1: Error reading Coverage for patient=%s: %s", patient_id, exc
        )
        return None


def _extract_payer_id(coverage: dict[str, Any]) -> str:
    """
    Extract payer identifier from Coverage.payor[0].

    Tries three sources in order:
      1. payor[0].identifier.value  (NPI or payer-specific ID — preferred)
      2. payor[0].reference         (FHIR Organization reference)
      3. payor[0].display           (human-readable name — last resort)
    """
    payors = coverage.get("payor", [])
    if not payors:
        return "UNKNOWN_PAYER"

    payor = payors[0]

    identifiers = payor.get("identifier", [])
    if isinstance(identifiers, dict):
        return identifiers.get("value", "UNKNOWN_PAYER")
    if isinstance(identifiers, list) and identifiers:
        return identifiers[0].get("value", "UNKNOWN_PAYER")

    ref = payor.get("reference", "")
    if ref:
        return ref.split("/")[-1]

    return payor.get("display", "UNKNOWN_PAYER")


# ---------------------------------------------------------------------------
# CRD endpoint caller
# ---------------------------------------------------------------------------

async def _call_crd_endpoint(
    payer_id: str,
    patient_id: str,
    cpt_code: str,
    coverage: dict[str, Any],
    config: CDSSConfig,
    encounter_id: Optional[str],
    practitioner_id: Optional[str],
) -> dict[str, Any]:
    """
    Call the payer's CDS Hooks CRD endpoint with an order-sign hook.

    The payer returns a list of cards. We look for:
      - PA-required language in card summary or detail
      - SMART app / DTR links (presence = PA required)
      - Critical indicator cards
      - Existing auth numbers in systemActions

    Returns dict with keys: status, auth_number_hint, raw_response
    """
    crd_url = config.payer_endpoints.get(payer_id)

    if not crd_url:
        logger.warning(
            "PA-1: No CRD endpoint configured for payer=%s. "
            "Returning UNKNOWN.",
            payer_id,
        )
        return {"status": PAStatus.UNKNOWN, "raw_response": None}

    hook_payload = _build_cds_hooks_payload(
        patient_id=patient_id,
        cpt_code=cpt_code,
        coverage=coverage,
        encounter_id=encounter_id,
        practitioner_id=practitioner_id,
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{crd_url}" if crd_url.endswith("/crd") else f"{crd_url}/order-sign",
                json=hook_payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if not resp.ok:
                    logger.error(
                        "PA-1: CRD endpoint returned %s for payer=%s",
                        resp.status, payer_id,
                    )
                    return {"status": PAStatus.UNKNOWN, "raw_response": None}

                crd_response = await resp.json()

    except aiohttp.ClientError as exc:
        logger.error("PA-1: CRD call failed for payer=%s: %s", payer_id, exc)
        return {"status": PAStatus.UNKNOWN, "raw_response": None}

    status, auth_hint = _parse_crd_cards(crd_response)
    return {
        "status": status,
        "auth_number_hint": auth_hint,
        "raw_response": crd_response,
    }


def _build_cds_hooks_payload(
    patient_id: str,
    cpt_code: str,
    coverage: dict[str, Any],
    encounter_id: Optional[str],
    practitioner_id: Optional[str],
) -> dict[str, Any]:
    """
    Construct a CDS Hooks order-sign hook payload.

    The fhirContext carries the proposed service as a draft ServiceRequest.
    Reference: https://build.fhir.org/ig/HL7/davinci-crd/hooks.html
    """
    service_request: dict[str, Any] = {
        "resourceType": "ServiceRequest",
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
        "insurance": [
            {"reference": f"Coverage/{coverage.get('id', 'unknown')}"}
        ],
    }

    if encounter_id:
        service_request["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    if practitioner_id:
        service_request["requester"] = {
            "reference": f"Practitioner/{practitioner_id}"
        }

    return {
        "hookInstance": f"pa-check-{patient_id}-{cpt_code}",
        "hook": "order-sign",
        "context": {
            "userId": f"Practitioner/{practitioner_id or 'unknown'}",
            "patientId": patient_id,
            "encounterId": encounter_id or "",
            "draftOrders": {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [{"resource": service_request}],
            },
        },
        "prefetch": {
            "coverage": coverage,
            "patient": {"resourceType": "Patient", "id": patient_id},
        },
    }


def _parse_crd_cards(
    crd_response: dict[str, Any],
) -> tuple[PAStatus, Optional[str]]:
    """
    Parse CDS Hooks cards to determine PA status.

    Payers signal PA requirement three ways:
      1. Card text containing PA-required language
      2. SMART / DTR link in card links (presence = PA required)
      3. Critical indicator card

    Returns (PAStatus, auth_number_hint)
    """
    cards = crd_response.get("cards", [])
    if not cards:
        return PAStatus.NOT_REQUIRED, None

    PA_REQUIRED_KEYWORDS = [
        "prior authorization required",
        "authorization required",
        "pa required",
        "precertification required",
    ]
    NO_PA_KEYWORDS = [
        "no prior authorization",
        "authorization not required",
        "pa not required",
        "not required",
    ]

    pa_required = False
    auth_hint = None

    for card in cards:
        summary = card.get("summary", "").lower()
        detail = card.get("detail", "").lower()
        indicator = card.get("indicator", "").lower()

        if any(kw in summary or kw in detail for kw in NO_PA_KEYWORDS):
            logger.info("PA-1: Card indicates PA NOT required")
            return PAStatus.NOT_REQUIRED, None

        if any(kw in summary or kw in detail for kw in PA_REQUIRED_KEYWORDS):
            pa_required = True
            logger.info("PA-1: Card indicates PA REQUIRED (indicator=%s)", indicator)

        # SMART/DTR link = PA required
        for link in card.get("links", []):
            if (
                link.get("type") == "smart"
                or "questionnaire" in link.get("url", "").lower()
            ):
                pa_required = True
                logger.info("PA-1: DTR/SMART link found — PA REQUIRED")

        # Check systemActions for existing auth number
        for action in card.get("systemActions", []):
            resource = action.get("resource", {})
            if resource.get("resourceType") == "ClaimResponse":
                auth_hint = _extract_auth_hint(resource)

        # Critical indicator with no other signal = treat as REQUIRED (safe default)
        if indicator == "critical" and not pa_required:
            pa_required = True
            logger.warning(
                "PA-1: Critical indicator card — defaulting to PA_REQUIRED"
            )

    return (PAStatus.REQUIRED, auth_hint) if pa_required else (PAStatus.NOT_REQUIRED, None)


def _extract_auth_hint(claim_response: dict[str, Any]) -> Optional[str]:
    """Extract a pre-existing auth number from a ClaimResponse systemAction."""
    for item in claim_response.get("item", []):
        for adj in item.get("adjudication", []):
            code = adj.get("category", {}).get("coding", [{}])[0].get("code", "")
            if code == "auth-ref":
                return adj.get("value", {}).get("value")
    return None


# ---------------------------------------------------------------------------
# Availity fallback
# ---------------------------------------------------------------------------

async def _availity_fallback(
    patient_id: str,
    cpt_code: str,
    config: CDSSConfig,
) -> CoverageCheckResult:
    """
    Fallback when no FHIR Coverage resource exists in the store.

    In production this calls the Availity eligibility API to determine
    payer identity and PA requirements for non-FHIR payers.

    Current implementation returns UNKNOWN and signals the agent to
    write a Task prompting clinical staff to verify insurance.

    TODO: Implement Availity /eligibility OAuth2 call using:
          config.availity_client_id_secret
          config.availity_client_secret_secret
    """
    logger.warning(
        "PA-1: Availity fallback — Coverage missing for patient=%s CPT=%s",
        patient_id, cpt_code,
    )

    return CoverageCheckResult(
        status=PAStatus.UNKNOWN,
        payer_id="UNKNOWN_PAYER",
        coverage_fhir_id=None,
        used_fallback=True,
        error_message=(
            "Coverage resource not found in FHIR store. "
            "Availity eligibility lookup required. "
            "Please verify patient insurance information."
        ),
    )