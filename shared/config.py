"""
shared/config.py
================
Central configuration for the ADK Prior Authorization Agent.

All sensitive values are loaded from Google Cloud Secret Manager
at runtime — never hardcoded here. Non-sensitive defaults can be
overridden via environment variables or your .env file.

Usage:
    from shared.config import get_config
    cfg = get_config()
    print(cfg.gcp_project_id)   # "healthcare-sa-2026"
    print(cfg.fhir_base_url)    # full Healthcare API base URL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(name: str, default: str = "") -> str:
    """Return env var or a safe default."""
    return os.environ.get(name, default)


@dataclass
class CDSSConfig:

    # ── GCP project ───────────────────────────────────────────────────────────
    gcp_project_id: str = field(
        default_factory=lambda: _env("GCP_PROJECT_ID", "healthcare-sa-2026")
    )
    gcp_region: str = field(
        default_factory=lambda: _env("GCP_REGION", "us-central1")
    )

    # ── FHIR store ────────────────────────────────────────────────────────────
    fhir_dataset: str = field(
        default_factory=lambda: _env("FHIR_DATASET", "cdss-dataset")
    )
    fhir_datastore: str = field(
        default_factory=lambda: _env("FHIR_DATASTORE", "cdss-fhir-store")
    )
    fhir_location: str = field(
        default_factory=lambda: _env("FHIR_LOCATION", "us-central1")
    )

    @property
    def fhir_base_url(self) -> str:
        """
        Full Healthcare API FHIR endpoint URL.
        Example:
          https://healthcare.googleapis.com/v1/projects/healthcare-sa-2026/
          locations/us-central1/datasets/cdss-dataset/fhirStores/cdss-fhir-store/fhir
        """
        return (
            f"https://healthcare.googleapis.com/v1"
            f"/projects/{self.gcp_project_id}"
            f"/locations/{self.fhir_location}"
            f"/datasets/{self.fhir_dataset}"
            f"/fhirStores/{self.fhir_datastore}/fhir"
        )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash")
    )
    # Keep temperature low — deterministic clinical output
    gemini_temperature: float = 0.1
    gemini_max_output_tokens: int = 8192

    # ── Payer endpoints ───────────────────────────────────────────────────────
    # Populated at runtime from Secret Manager by agent.py
    # Structure: { "payer_id": "https://payer-crd-endpoint.com/cds-services" }
    # Register payers using: scripts/setup_payer_secrets.py
    payer_endpoints: dict = field(default_factory=dict)

    # Availity — fallback for payers without Da Vinci FHIR API support
    availity_base_url: str = field(
        default_factory=lambda: _env(
            "AVAILITY_BASE_URL", "https://api.availity.com/availity/v1"
        )
    )
    availity_client_id_secret: str = "availity-client-id"
    availity_client_secret_secret: str = "availity-client-secret"

    # ── Firestore cache ───────────────────────────────────────────────────────
    # Payer questionnaires cached to avoid repeated payer API calls
    # Cache key pattern: pa-q--{payer_id}--{cpt_code}
    firestore_collection_questionnaires: str = "pa_questionnaire_cache"
    firestore_collection_pa_status: str = "pa_submission_status"
    dtr_cache_ttl_hours: int = field(
        default_factory=lambda: int(_env("DTR_CACHE_TTL_HOURS", "24"))
    )

    # ── Pub/Sub ───────────────────────────────────────────────────────────────
    # orchestrator-ready → upstream pipeline publishes when FHIR record ready
    # prior-auth-ready   → PA agent publishes when ClaimResponse received
    pubsub_topic_orchestrator_ready: str = "orchestrator-ready"
    pubsub_topic_prior_auth_ready: str = "prior-auth-ready"
    pubsub_subscription_pa_agent: str = "prior-auth-agent-sub"

    # ── Cloud DLP ─────────────────────────────────────────────────────────────
    # Applied before every outbound PHI transmission to the payer.
    # PA Agent is the only agent in this suite that sends PHI externally.
    doc_agent_dlp_template: str = field(
        default_factory=lambda: _env("DLP_TEMPLATE", "cdss-phi-doc-agents")
    )

    @property
    def dlp_inspect_template_name(self) -> str:
        return (
            f"projects/{self.gcp_project_id}"
            f"/inspectTemplates/{self.doc_agent_dlp_template}"
        )

    # ── PA polling windows (CMS-0057-F) ──────────────────────────────────────
    # Expedited: 72 hours  |  Standard: 7 days (168 hours)
    pas_poll_interval_minutes: int = field(
        default_factory=lambda: int(_env("PAS_POLL_INTERVAL_MINUTES", "15"))
    )
    pas_max_poll_hours_expedited: int = 72
    pas_max_poll_hours_standard: int = 168

    # ── Secret Manager key names (IDs only — not the values) ─────────────────
    secret_payer_endpoints: str = "pa-payer-endpoints"
    secret_gemini_api_key: str = "gemini-api-key"

    # ── Other agents (defined here so entire suite shares one config) ─────────
    cdi_confidence_threshold: float = field(
        default_factory=lambda: float(_env("CDI_CONFIDENCE_THRESHOLD", "0.75"))
    )
    handoff_schedule_times: list = field(
        default_factory=lambda: ["06:30", "18:30"]
    )


@lru_cache(maxsize=1)
def get_config() -> CDSSConfig:
    """
    Return the singleton CDSSConfig instance.
    lru_cache means this is constructed once per process.

    Example:
        from shared.config import get_config
        cfg = get_config()
    """
    return CDSSConfig()