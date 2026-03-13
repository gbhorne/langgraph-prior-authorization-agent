"""
agents/prior_auth/tools/questionnaire_filler.py
================================================
PA-3: Gemini Questionnaire Auto-Population

This is the core AI step of the Prior Authorization Agent.

What this step does:
  1. Assembles patient FHIR resources into a clinical context
  2. Sends Questionnaire items + FHIR context to Gemini 2.5 Flash
  3. Receives Gemini's answers as structured JSON
  4. VALIDATES every answer:
       - Cited resource IDs must exist in the actual FHIR bundle
       - Required questions with no evidence → MISSING (never guessed)
       - Confidence values converted to AnswerConfidence enum
  5. Returns validated list of QuestionnaireAnswer objects

Anti-hallucination design:
  - MISSING is the only valid answer when no FHIR evidence exists
  - Post-generation validation checks every cited resource ID
  - Invalid citations → downgraded to LOW confidence + flagged
  - The prompt repeats the citation rule three times intentionally
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from agents.prior_auth.prompts import build_questionnaire_prompt
from shared.config import CDSSConfig, get_config
from shared.fhir_client import FHIRClient
from shared.models import AnswerConfidence, EvidenceSource, QuestionnaireAnswer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def fill_questionnaire(
    questionnaire: dict[str, Any],
    patient_bundle: dict[str, Any],
    clinical_impression: dict[str, Any],
    fhir_client: FHIRClient,
    config: Optional[CDSSConfig] = None,
) -> list[QuestionnaireAnswer]:
    """
    Auto-populate a payer questionnaire using Gemini + patient FHIR data.

    Args:
        questionnaire:       FHIR Questionnaire resource from PA-2
        patient_bundle:      Patient $everything bundle from FHIR store
        clinical_impression: ClinicalImpression resource from upstream pipeline
        fhir_client:         Initialized FHIRClient (used for citation validation)
        config:              CDSSConfig (defaults to singleton)

    Returns:
        List of QuestionnaireAnswer objects — one per questionnaire item.
        Required items with no evidence have confidence=MISSING.
        These block submission and generate a Task instead.
    """
    cfg = config or get_config()

    # Extract FHIR resources from bundle
    resources = _extract_resources_from_bundle(patient_bundle)

    # Build prompts
    system_prompt, user_prompt = build_questionnaire_prompt(
        questionnaire_items=questionnaire.get("item", []),
        clinical_impression=clinical_impression,
        conditions=resources["conditions"],
        observations=resources["observations"],
        medications=resources["medications"],
        diagnostic_reports=resources["diagnostic_reports"],
        allergies=resources["allergies"],
    )

    # Call Gemini
    raw_answers = await _call_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=cfg,
    )

    # Build resource ID index for citation validation
    resource_id_index = _build_resource_id_index(patient_bundle)

    # Validate and convert answers
    validated = _validate_answers(
        raw_answers=raw_answers,
        questionnaire_items=questionnaire.get("item", []),
        resource_id_index=resource_id_index,
    )

    missing_count = sum(
        1 for a in validated
        if a.is_required and a.confidence == AnswerConfidence.MISSING
    )

    logger.info(
        "PA-3: Questionnaire filled — %d answers, %d required MISSING",
        len(validated), missing_count,
    )

    return validated


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

async def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    config: CDSSConfig,
) -> list[dict[str, Any]]:
    """
    Call Gemini 2.5 Flash and parse the JSON response.

    Uses Vertex AI SDK. Project and region from config.
    Returns the parsed JSON array of answer dicts.

    If Gemini returns malformed JSON, logs the error and returns
    an empty list — the agent will treat all questions as MISSING
    and write a Task for clinician review.
    """
    import asyncio

    loop = asyncio.get_event_loop()

    def _sync_call() -> list[dict[str, Any]]:
        try:
            vertexai.init(
                project=config.gcp_project_id,
                location=config.gcp_region,
            )

            model = GenerativeModel(
                model_name=config.gemini_model,
                system_instruction=system_prompt,
            )

            generation_config = GenerationConfig(
                temperature=config.gemini_temperature,
                max_output_tokens=config.gemini_max_output_tokens,
                response_mime_type="application/json",
            )

            response = model.generate_content(
                contents=user_prompt,
                generation_config=generation_config,
            )

            raw_text = response.text.strip()

            # Strip markdown fences if Gemini added them despite instructions
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                ).strip()

            parsed = json.loads(raw_text)

            if not isinstance(parsed, list):
                logger.error(
                    "PA-3: Gemini returned non-list JSON: %s",
                    type(parsed).__name__,
                )
                return []

            logger.info(
                "PA-3: Gemini returned %d answers", len(parsed)
            )
            return parsed

        except json.JSONDecodeError as exc:
            logger.error("PA-3: Gemini JSON parse failed: %s", exc)
            return []
        except Exception as exc:
            logger.error("PA-3: Gemini call failed: %s", exc)
            return []

    return await loop.run_in_executor(None, _sync_call)


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def _build_resource_id_index(bundle: dict[str, Any]) -> set[str]:
    """
    Build a set of all resource IDs in the patient bundle.

    Used to validate that Gemini's cited resource IDs are real.
    Hallucinated IDs will not be in this set.
    """
    ids: set[str] = set()
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rid = resource.get("id")
        if rid:
            ids.add(rid)
    return ids


def _validate_answers(
    raw_answers: list[dict[str, Any]],
    questionnaire_items: list[dict[str, Any]],
    resource_id_index: set[str],
) -> list[QuestionnaireAnswer]:
    """
    Validate Gemini's raw answers and convert to QuestionnaireAnswer objects.

    Validation rules:
      1. Cited resource ID must exist in resource_id_index
         → If invalid: downgrade confidence to LOW, flag for review
      2. Non-MISSING answers must have a resource ID
         → If missing ID: downgrade to MISSING
      3. Confidence values must be valid AnswerConfidence enum members
         → If invalid: default to LOW

    Also cross-references questionnaire items to mark is_required correctly.
    """
    # Build required question index from questionnaire
    required_link_ids = _get_required_link_ids(questionnaire_items)

    validated: list[QuestionnaireAnswer] = []

    for raw in raw_answers:
        link_id = raw.get("linkId", "")
        confidence_str = raw.get("confidence", "MISSING").upper()
        evidence_id = raw.get("evidence_resource_id")
        answer_value = raw.get("answer_value")
        is_required = link_id in required_link_ids

        # Validate confidence string
        try:
            confidence = AnswerConfidence(confidence_str)
        except ValueError:
            logger.warning(
                "PA-3: Invalid confidence '%s' for linkId=%s — defaulting to LOW",
                confidence_str, link_id,
            )
            confidence = AnswerConfidence.LOW

        # Validate citation
        if confidence != AnswerConfidence.MISSING:
            if not evidence_id:
                # Gemini provided an answer without a citation
                logger.warning(
                    "PA-3: Answer for linkId=%s has no evidence_resource_id "
                    "but confidence=%s — downgrading to MISSING",
                    link_id, confidence,
                )
                confidence = AnswerConfidence.MISSING
                answer_value = None
                evidence_id = None

            elif evidence_id not in resource_id_index:
                # Gemini hallucinated a resource ID
                logger.warning(
                    "PA-3: Hallucinated resource ID '%s' for linkId=%s — "
                    "downgrading to LOW confidence",
                    evidence_id, link_id,
                )
                confidence = AnswerConfidence.LOW
                # Keep the answer value but flag the citation as suspect

        # Build EvidenceSource if we have a valid citation
        evidence_sources: list[EvidenceSource] = []
        if evidence_id and evidence_id in resource_id_index:
            evidence_sources.append(
                EvidenceSource(
                    resource_type=_infer_resource_type(evidence_id),
                    resource_id=evidence_id,
                    value=str(answer_value) if answer_value is not None else None,
                )
            )

        try:
            answer = QuestionnaireAnswer(
                link_id=link_id,
                question_text=raw.get("question_text"),
                answer_value=answer_value,
                evidence_resource_id=evidence_id if confidence != AnswerConfidence.MISSING else None,
                evidence_sources=evidence_sources,
                confidence=confidence,
                missing_info_needed=raw.get("missing_info_needed"),
                is_required=is_required,
            )
            validated.append(answer)

        except Exception as exc:
            logger.error(
                "PA-3: Failed to construct QuestionnaireAnswer for linkId=%s: %s",
                link_id, exc,
            )
            # Create a safe MISSING answer so we don't lose the question
            validated.append(
                QuestionnaireAnswer(
                    link_id=link_id,
                    question_text=raw.get("question_text"),
                    answer_value=None,
                    evidence_resource_id=None,
                    confidence=AnswerConfidence.MISSING,
                    missing_info_needed=(
                        f"Answer construction failed: {exc}. "
                        f"Manual review required."
                    ),
                    is_required=is_required,
                )
            )

    # Check for questionnaire items that Gemini did not answer at all
    answered_link_ids = {a.link_id for a in validated}
    for item in questionnaire_items:
        link_id = item.get("linkId", "")
        if link_id and link_id not in answered_link_ids:
            logger.warning(
                "PA-3: Questionnaire item linkId=%s was not answered by Gemini",
                link_id,
            )
            validated.append(
                QuestionnaireAnswer(
                    link_id=link_id,
                    question_text=item.get("text"),
                    answer_value=None,
                    evidence_resource_id=None,
                    confidence=AnswerConfidence.MISSING,
                    missing_info_needed="This question was not addressed in the AI response.",
                    is_required=link_id in required_link_ids,
                )
            )

    return validated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_resources_from_bundle(
    bundle: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract and categorize FHIR resources from a $everything bundle.
    Returns a dict keyed by resource category for prompt building.
    """
    all_resources: list[dict[str, Any]] = [
        entry["resource"]
        for entry in bundle.get("entry", [])
        if "resource" in entry
    ]

    def of_type(rtype: str) -> list[dict[str, Any]]:
        return [r for r in all_resources if r.get("resourceType") == rtype]

    return {
        "conditions":        of_type("Condition"),
        "observations":      of_type("Observation"),
        "medications":       of_type("MedicationRequest"),
        "diagnostic_reports": of_type("DiagnosticReport"),
        "allergies":         of_type("AllergyIntolerance"),
        "coverage":          of_type("Coverage"),
        "encounters":        of_type("Encounter"),
        "procedures":        of_type("Procedure"),
    }


def _get_required_link_ids(items: list[dict[str, Any]]) -> set[str]:
    """
    Extract linkIds of required questionnaire items.
    Handles nested items recursively.
    """
    required: set[str] = set()

    def _recurse(item_list: list[dict[str, Any]]) -> None:
        for item in item_list:
            link_id = item.get("linkId", "")
            # FHIR Questionnaire required flag
            if item.get("required", False):
                required.add(link_id)
            # Recurse into nested items
            sub_items = item.get("item", [])
            if sub_items:
                _recurse(sub_items)

    _recurse(items)
    return required


def _infer_resource_type(resource_id: str) -> str:
    """
    Infer FHIR resource type from ID prefix conventions.

    Many FHIR implementations use prefixes in resource IDs:
      obs-*   → Observation
      cond-*  → Condition
      med-*   → MedicationRequest
      dr-*    → DiagnosticReport
      enc-*   → Encounter

    Falls back to "Unknown" if no prefix matches.
    This is best-effort for EvidenceSource population only.
    """
    prefix_map = {
        "obs": "Observation",
        "cond": "Condition",
        "med": "MedicationRequest",
        "dr": "DiagnosticReport",
        "enc": "Encounter",
        "proc": "Procedure",
        "allergy": "AllergyIntolerance",
        "coverage": "Coverage",
        "patient": "Patient",
        "pract": "Practitioner",
    }
    lower_id = resource_id.lower()
    for prefix, rtype in prefix_map.items():
        if lower_id.startswith(prefix):
            return rtype
    return "Unknown"