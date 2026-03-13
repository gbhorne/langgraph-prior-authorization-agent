"""
shared/models.py
================
Pydantic v2 data models shared across all five HC-CDSS document agents.

These are the data contracts between every agent step.
Using models instead of raw dicts eliminates silent key bugs
and makes the pipeline self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# Enumerations
# ===========================================================================

class PAStatus(str, Enum):
    """Outcome of the PA-1 coverage check."""
    REQUIRED = "PA_REQUIRED"
    NOT_REQUIRED = "PA_NOT_REQUIRED"
    UNKNOWN = "UNKNOWN"


class AnswerConfidence(str, Enum):
    """
    Confidence of a Gemini-generated questionnaire answer (PA-3).

    HIGH     — direct match to a FHIR resource value
    MODERATE — reasonable inference from available data
    LOW      — weak evidence; clinician should verify
    MISSING  — no supporting evidence; answer left blank
    """
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MISSING = "MISSING"


class ClaimDecision(str, Enum):
    """Payer decision from a ClaimResponse resource."""
    APPROVED = "approved"
    DENIED = "denied"
    PENDED = "pended"       # Payer needs more info
    PENDING = "pending"     # Awaiting payer response (our internal state)
    ERROR = "error"


class GapPriority(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class GapAssignee(str, Enum):
    CLINICIAN = "CLINICIAN"
    CDI_SPECIALIST = "CDI_SPECIALIST"


class IllnessSeverity(str, Enum):
    STABLE = "STABLE"
    WATCHER = "WATCHER"
    UNSTABLE = "UNSTABLE"


class ReferralUrgency(str, Enum):
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    EMERGENT = "EMERGENT"


# ===========================================================================
# FHIR identity helpers
# ===========================================================================

class FHIRReference(BaseModel):
    """Lightweight reference to a FHIR resource."""
    resource_type: str = Field(
        ..., examples=["Observation", "Condition", "MedicationRequest"]
    )
    resource_id: str = Field(..., examples=["obs-12345"])
    display: Optional[str] = None

    @property
    def reference_string(self) -> str:
        """Returns e.g. 'Observation/obs-12345'"""
        return f"{self.resource_type}/{self.resource_id}"


class FHIRResource(BaseModel):
    """A FHIR resource payload as returned by the Healthcare API."""
    resource_type: str
    resource_id: str
    resource: dict[str, Any] = Field(description="Raw FHIR resource JSON")


# ===========================================================================
# Citation models
# ===========================================================================

class EvidenceSource(BaseModel):
    """
    One piece of FHIR evidence supporting a clinical claim.
    PA-3 requires every non-MISSING answer to have at least one.
    """
    resource_type: str
    resource_id: str
    value: Optional[str] = Field(
        default=None,
        description="The specific value from this resource, e.g. '138/88 mmHg'"
    )
    effective_date: Optional[datetime] = None


class CitedClaim(BaseModel):
    """A clinical claim paired with its FHIR evidence source."""
    claim_text: str
    resource_type: str
    resource_id: str
    confidence: AnswerConfidence = AnswerConfidence.HIGH


# ===========================================================================
# PA Agent models (Steps PA-1 through PA-5)
# ===========================================================================

class QuestionnaireAnswer(BaseModel):
    """
    A single answer to one payer questionnaire item (PA-3 output).

    The citation validator below is the primary anti-hallucination guard.
    Gemini must cite a real FHIR resource ID or mark the answer MISSING.
    It cannot do both — and it cannot provide an answer without a citation.
    """
    link_id: str = Field(
        ..., description="Questionnaire.item[].linkId — identifies the question"
    )
    question_text: Optional[str] = None
    answer_value: Optional[Any] = Field(
        default=None,
        description="The answer. Type matches questionnaire item type."
    )
    evidence_resource_id: Optional[str] = Field(
        default=None,
        description="FHIR resource ID supporting this answer. "
                    "Required for HIGH/MODERATE/LOW. None only if MISSING."
    )
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    confidence: AnswerConfidence
    missing_info_needed: Optional[str] = Field(
        default=None,
        description="What clinical data is needed — only populated if MISSING"
    )
    is_required: bool = False

    @field_validator("evidence_resource_id", mode="after")
    @classmethod
    def citation_required_unless_missing(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        """
        Anti-hallucination guard:
        If confidence != MISSING, a real resource ID must be cited.
        """
        confidence = info.data.get("confidence")
        if confidence and confidence != AnswerConfidence.MISSING and not v:
            raise ValueError(
                f"evidence_resource_id is required when confidence='{confidence}'. "
                f"Provide a FHIR resource ID or set confidence=MISSING."
            )
        return v


class PATaskItem(BaseModel):
    """
    A missing documentation item that blocks PA submission.

    When required questions have confidence=MISSING, the agent
    writes a FHIR Task resource instead of submitting the bundle.
    Each PATaskItem becomes one note in that Task.
    """
    link_id: str
    question_text: str
    missing_info_needed: str
    suggested_source: Optional[str] = Field(
        default=None,
        description="Where the clinician might find this, e.g. 'Recent echo report'"
    )


class ClaimResponseDecision(BaseModel):
    """
    The payer's decision on a PA request (PA-5 output).
    Written to FHIR as a ClaimResponse and published to Pub/Sub.
    """
    claim_response_id: Optional[str] = None
    decision: ClaimDecision
    auth_number: Optional[str] = None
    denial_reason: Optional[str] = None
    pended_items: list[str] = Field(default_factory=list)
    decision_timestamp: Optional[datetime] = None
    poll_count: int = 0
    raw_claim_response: Optional[dict[str, Any]] = None


class PAAgentResult(BaseModel):
    """
    Top-level result returned by the PA Agent orchestrator.
    Complete audit record of one PA run.
    """
    patient_id: str
    encounter_id: Optional[str] = None
    cpt_code: str
    payer_id: str

    # Step outcomes
    pa_required: PAStatus = PAStatus.UNKNOWN
    questionnaire_id: Optional[str] = None
    answers: list[QuestionnaireAnswer] = Field(default_factory=list)
    missing_items: list[PATaskItem] = Field(default_factory=list)
    claim_fhir_id: Optional[str] = None
    questionnaire_response_fhir_id: Optional[str] = None
    decision: Optional[ClaimResponseDecision] = None

    # Submission state
    submission_id: Optional[str] = None
    blocked_by_missing: bool = False
    task_fhir_id: Optional[str] = None

    # Timing
    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def missing_required_count(self) -> int:
        """Count of required questions with MISSING confidence. Gates submission."""
        return sum(
            1 for a in self.answers
            if a.is_required and a.confidence == AnswerConfidence.MISSING
        )


# ===========================================================================
# Document output model (Discharge Summary, Referral, Handoff agents)
# ===========================================================================

class DocumentReferenceOutput(BaseModel):
    """
    Wraps a generated clinical document before writing to FHIR.
    doc_status=preliminary until a clinician signs → promotes to final.
    """
    loinc_code: str = Field(
        ..., examples=["18842-5", "57133-1", "34109-9", "52521-2"]
    )
    loinc_display: str
    content_text: str
    citations: list[CitedClaim] = Field(default_factory=list)
    uncited_claims: list[str] = Field(
        default_factory=list,
        description="Claims flagged for clinician review — no citation found"
    )
    status: str = "current"
    doc_status: str = "preliminary"
    fhir_resource_id: Optional[str] = None


# ===========================================================================
# CDI Agent models (Agent 5)
# ===========================================================================

class GapFinding(BaseModel):
    """
    A CDI documentation gap. Never directly modifies FHIR —
    only flags for human action via a Task resource.
    """
    gap_type: str = Field(
        ...,
        examples=[
            "specificity_gap", "undocumented_comorbidity",
            "procedure_diagnosis_mismatch", "em_level_underdocumented",
            "present_on_admission_missing"
        ]
    )
    current_code: Optional[str] = None
    suggested_code: Optional[str] = None
    rationale: str
    supporting_evidence: list[EvidenceSource] = Field(default_factory=list)
    priority: GapPriority = GapPriority.MODERATE
    assigned_to: GapAssignee = GapAssignee.CDI_SPECIALIST
    estimated_drg_impact: Optional[str] = None


# ===========================================================================
# Handoff Agent models (Agent 4)
# ===========================================================================

class HandoffItem(BaseModel):
    """
    One action or situation-awareness item in an I-PASS handoff.
    Action items need owner_role + timeframe.
    Situation awareness items need condition + action (if/then).
    """
    item_type: str = Field(
        ..., description="'action' or 'situation_awareness'"
    )
    description: str
    owner_role: Optional[str] = None
    timeframe: Optional[str] = None
    condition: Optional[str] = None   # "If" clause
    action: Optional[str] = None      # "Then" clause
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None