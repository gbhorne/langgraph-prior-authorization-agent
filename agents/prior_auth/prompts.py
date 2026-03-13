"""
agents/prior_auth/prompts.py
============================
All Gemini prompt templates for the Prior Authorization Agent.

Design principles:
  1. SYSTEM prompt is immutable — defines clinical persona and rules.
     Never allow user input to modify it.
  2. CONTEXT section is built from validated FHIR data only.
  3. OUTPUT is strict JSON — no preamble, no markdown, raw JSON only.
  4. The citation rule is stated three times intentionally —
     it is the most important constraint in the entire agent.
  5. MISSING is the only valid answer when no FHIR evidence exists.
     The agent never invents clinical data.

Usage:
    from agents.prior_auth.prompts import build_questionnaire_prompt

    system, user = build_questionnaire_prompt(
        questionnaire_items=items,
        clinical_impression=impression_dict,
        conditions=conditions_list,
        observations=observations_list,
        medications=medications_list,
        diagnostic_reports=reports_list,
    )
"""

from __future__ import annotations

import json
from typing import Any


# ===========================================================================
# PA-3 — Questionnaire auto-population
# ===========================================================================

PA_QUESTIONNAIRE_SYSTEM = """You are a prior authorization documentation specialist \
working within a clinical decision support system.

Your role is to populate a payer's prior authorization questionnaire using ONLY \
the clinical data provided in the FHIR resources below.

ABSOLUTE RULES — follow without exception:

1. CITATION REQUIRED: Every answer must include an evidence_resource_id — \
the exact FHIR resource ID that directly supports your answer.

2. NEVER INVENT: If no FHIR resource supports an answer, set confidence to \
"MISSING" and leave answer_value as null. Do not guess. Do not infer beyond \
what the data explicitly shows.

3. CONFIDENCE LEVELS:
   HIGH     — direct, unambiguous value in a FHIR resource
   MODERATE — reasonable interpretation of available data
   LOW      — weak or indirect evidence; clinician should verify
   MISSING  — no supporting evidence exists — DO NOT guess

4. REQUIRED QUESTIONS: If a required question has confidence MISSING, the \
submission will be blocked and sent for clinician review. This is correct \
behavior — a bad submission causes more harm than a delayed one.

5. OUTPUT FORMAT: Return ONLY a JSON array. No preamble. No explanation. \
No markdown code fences. Raw JSON only."""


def build_questionnaire_prompt(
    questionnaire_items: list[dict[str, Any]],
    clinical_impression: dict[str, Any],
    conditions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    medications: list[dict[str, Any]],
    diagnostic_reports: list[dict[str, Any]],
    allergies: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """
    Build the system + user prompt pair for PA-3 questionnaire filling.

    Args:
        questionnaire_items:  Questionnaire.item[] array from the payer
        clinical_impression:  ClinicalImpression FHIR resource dict
        conditions:           List of Condition resource dicts
        observations:         List of Observation resource dicts
        medications:          List of MedicationRequest resource dicts
        diagnostic_reports:   List of DiagnosticReport resource dicts
        allergies:            Optional list of AllergyIntolerance dicts

    Returns:
        (system_prompt, user_prompt) — pass both to the Gemini API call
    """
    allergies = allergies or []

    context = _build_context_block(
        clinical_impression=clinical_impression,
        conditions=conditions,
        observations=observations,
        medications=medications,
        diagnostic_reports=diagnostic_reports,
        allergies=allergies,
    )
    task = _build_task_block(questionnaire_items)
    output_spec = _build_output_spec()

    user_prompt = f"{context}\n\n{task}\n\n{output_spec}"
    return PA_QUESTIONNAIRE_SYSTEM, user_prompt


def _build_context_block(
    clinical_impression: dict[str, Any],
    conditions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    medications: list[dict[str, Any]],
    diagnostic_reports: list[dict[str, Any]],
    allergies: list[dict[str, Any]],
) -> str:
    """Serialize FHIR resources into the CONTEXT section of the prompt."""
    impression_summary = _summarize_clinical_impression(clinical_impression)
    allergy_section = (
        _serialize_resources(allergies, "AllergyIntolerance")
        if allergies
        else "None documented in record."
    )

    return f"""=== CLINICAL CONTEXT (FHIR DATA) ===

--- CLINICAL IMPRESSION ---
{impression_summary}
ClinicalImpression resource ID: {clinical_impression.get('id', 'UNKNOWN')}

--- CONDITIONS ---
{_serialize_resources(conditions, 'Condition')}

--- OBSERVATIONS (Labs and Vitals) ---
{_serialize_resources(observations, 'Observation')}

--- MEDICATIONS ---
{_serialize_resources(medications, 'MedicationRequest')}

--- DIAGNOSTIC REPORTS ---
{_serialize_resources(diagnostic_reports, 'DiagnosticReport')}

--- ALLERGIES ---
{allergy_section}

IMPORTANT: The "id" fields above (e.g. "id": "obs-12345") are the values
you MUST use as evidence_resource_id. Do not fabricate IDs."""


def _build_task_block(questionnaire_items: list[dict[str, Any]]) -> str:
    """Serialize questionnaire items as the TASK section."""
    items_json = json.dumps(questionnaire_items, indent=2)
    return f"""=== TASK: POPULATE PRIOR AUTHORIZATION QUESTIONNAIRE ===

For each item below, answer using ONLY the FHIR data in the CLINICAL CONTEXT.
Cite a real resource ID for every non-MISSING answer.
Set confidence=MISSING if no resource supports the answer.
Never invent clinical data.

QUESTIONNAIRE ITEMS:
{items_json}"""


def _build_output_spec() -> str:
    """Return the strict JSON output format specification."""
    return """=== OUTPUT FORMAT ===

Return ONLY a JSON array. No markdown. No explanation. Raw JSON only.

Schema per element:
{
  "linkId": "<exact linkId from the questionnaire item>",
  "question_text": "<question text for readability>",
  "answer_value": <answer or null if MISSING>,
  "evidence_resource_id": "<FHIR resource ID — required unless MISSING>",
  "confidence": "<HIGH|MODERATE|LOW|MISSING>",
  "missing_info_needed": "<what is needed — only if MISSING, else null>",
  "is_required": <true|false>
}

Example HIGH confidence answer:
{
  "linkId": "Q1",
  "question_text": "Primary diagnosis?",
  "answer_value": "Type 2 diabetes mellitus without complications",
  "evidence_resource_id": "cond-78901",
  "confidence": "HIGH",
  "missing_info_needed": null,
  "is_required": true
}

Example MISSING answer:
{
  "linkId": "Q7",
  "question_text": "Pulmonary function test in last 12 months?",
  "answer_value": null,
  "evidence_resource_id": null,
  "confidence": "MISSING",
  "missing_info_needed": "No PFT DiagnosticReport found in the patient record.",
  "is_required": true
}"""


# ===========================================================================
# PA-5 — Pended decision analysis
# ===========================================================================

PA_PENDED_TASK_SYSTEM = """You are a clinical documentation specialist.
The payer returned a 'pended' prior authorization — they need additional
information before making a final decision.

Map each payer-requested item to clinical documentation in the patient's
FHIR record, or identify what is genuinely missing.

Return ONLY JSON. No preamble. No explanation."""


def build_pended_task_prompt(
    pended_items: list[str],
    patient_bundle: dict[str, Any],
) -> tuple[str, str]:
    """
    Build the prompt to analyze payer pended items.

    Args:
        pended_items:    Items the payer flagged as needed
        patient_bundle:  Patient FHIR $everything bundle

    Returns:
        (system_prompt, user_prompt)
    """
    pended_json = json.dumps(pended_items, indent=2)
    bundle_summary = _summarize_bundle_for_pended(patient_bundle)

    user_prompt = f"""The payer needs the following additional information:

PAYER PENDED ITEMS:
{pended_json}

AVAILABLE PATIENT RECORD:
{bundle_summary}

For each pended item return:
{{
  "pended_item": "<the payer's requested item>",
  "status": "EXISTS_IN_RECORD | NEEDS_FORMATTING | GENUINELY_MISSING",
  "resource_id": "<FHIR resource ID if exists, else null>",
  "resource_type": "<FHIR resource type if exists, else null>",
  "action_required": "<specific action for clinician or CDI team>",
  "suggested_documentation": "<what to submit to the payer>"
}}"""

    return PA_PENDED_TASK_SYSTEM, user_prompt


# ===========================================================================
# Urgency classification
# ===========================================================================

PA_URGENCY_SYSTEM = """You are a prior authorization coordinator.
Determine if this PA request should be STANDARD or EXPEDITED.

EXPEDITED criteria (any one is sufficient):
- Life-threatening condition
- Serious risk of permanent impairment
- Unable to reasonably obtain standard PA given the clinical urgency
- ClinicalImpression contains high-severity indicators

Return ONLY JSON. No preamble."""


def build_urgency_prompt(
    clinical_impression: dict[str, Any],
    care_plan: dict[str, Any],
    detected_issues: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Build the prompt to classify PA urgency (STANDARD vs EXPEDITED).
    EXPEDITED triggers the 72-hour CMS window instead of 7-day standard.
    """
    impression_summary = _summarize_clinical_impression(clinical_impression)
    issues_summary = (
        json.dumps(detected_issues, indent=2)
        if detected_issues
        else "No drug interactions or safety alerts flagged."
    )

    user_prompt = f"""Classify the urgency of this prior authorization request.

CLINICAL IMPRESSION:
{impression_summary}

CARE PLAN (proposed service):
{json.dumps(care_plan, indent=2)}

DETECTED ISSUES:
{issues_summary}

Return JSON:
{{
  "urgency": "STANDARD | EXPEDITED",
  "rationale": "<one sentence explaining the classification>",
  "expedited_criteria_met": ["<criteria met — empty list if STANDARD>"]
}}"""

    return PA_URGENCY_SYSTEM, user_prompt


# ===========================================================================
# Private helpers
# ===========================================================================

def _summarize_clinical_impression(impression: dict[str, Any]) -> str:
    """Extract key fields from ClinicalImpression for prompt injection."""
    if not impression:
        return "No ClinicalImpression available."

    lines = [
        f"Status: {impression.get('status', 'unknown')}",
        f"Resource ID: {impression.get('id', 'UNKNOWN')}",
    ]

    description = impression.get("description", "")
    if description:
        lines.append(f"Summary: {description}")

    findings = impression.get("finding", [])
    if findings:
        lines.append("Findings:")
        for f in findings[:10]:
            item = f.get("itemCodeableConcept", {})
            codes = item.get("coding", [])
            display = item.get("text", "")
            if codes:
                code_str = ", ".join(
                    f"{c.get('code', '')} ({c.get('display', '')})"
                    for c in codes
                )
                lines.append(f"  - {code_str} | {display}")

    return "\n".join(lines)


def _serialize_resources(
    resources: list[dict[str, Any]],
    resource_type: str,
    max_resources: int = 20,
) -> str:
    """
    Serialize FHIR resources to compact JSON for prompt injection.
    Caps at max_resources to avoid context window overflow.
    """
    if not resources:
        return f"No {resource_type} resources in patient record."

    truncation_note = ""
    if len(resources) > max_resources:
        resources = resources[:max_resources]
        truncation_note = (
            f"\n[Showing {max_resources} most recent {resource_type} resources only]"
        )

    return json.dumps(resources, indent=2, default=str) + truncation_note


def _summarize_bundle_for_pended(bundle: dict[str, Any]) -> str:
    """
    Summarize a $everything bundle as resource type counts + IDs.
    Used for pended task analysis — avoids dumping the full bundle.
    """
    entries = bundle.get("entry", [])
    type_counts: dict[str, int] = {}
    type_ids: dict[str, list[str]] = {}

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "Unknown")
        rid = resource.get("id", "")
        type_counts[rtype] = type_counts.get(rtype, 0) + 1
        type_ids.setdefault(rtype, []).append(rid)

    lines = ["Available resources in patient record:"]
    for rtype, count in sorted(type_counts.items()):
        ids_preview = ", ".join(type_ids[rtype][:3])
        if count > 3:
            ids_preview += f" ... (+{count - 3} more)"
        lines.append(f"  {rtype}: {count} (IDs: {ids_preview})")

    return "\n".join(lines)