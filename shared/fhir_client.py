"""
shared/fhir_client.py
=====================
Async wrapper around the Google Cloud Healthcare FHIR R4 REST API.

Every agent in this suite uses this client to read from and write
to the FHIR store. Handles authentication, retries, and pagination.

Local dev setup (VS Code):
    1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
    2. Run in terminal: gcloud auth application-default login
    3. Run in terminal: gcloud config set project healthcare-sa-2026
    The client picks up ADC automatically — no key files needed.

Production (Cloud Run):
    Attach the service account with Healthcare FHIR Editor role.
    ADC resolves via the metadata server automatically.

Usage:
    from shared.fhir_client import FHIRClient
    from shared.config import get_config

    async with FHIRClient(get_config()) as client:
        patient    = await client.read("Patient", "patient-123")
        conditions = await client.search("Condition", {"patient": "patient-123"})
        written    = await client.create("DocumentReference", doc_dict)
        bundle     = await client.everything("patient-123")
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest

from shared.config import CDSSConfig, get_config

logger = logging.getLogger(__name__)

# Retry config
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.0


class FHIRClient:
    """
    Async FHIR R4 client for the Google Cloud Healthcare API.

    Instantiate once per agent run — do not create a new client
    per request. Use as an async context manager for clean teardown.

    Example:
        async with FHIRClient(cfg) as client:
            patient = await client.read("Patient", "patient-123")
    """

    def __init__(self, config: Optional[CDSSConfig] = None):
        self._config = config or get_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._credentials = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "FHIRClient":
        await self._init_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _init_session(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/fhir+json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Authentication ────────────────────────────────────────────────────────

    def _get_access_token(self) -> str:
        """
        Get a valid Google OAuth2 access token via Application Default Credentials.
        Local dev: resolved from gcloud auth application-default login.
        Cloud Run: resolved from the attached service account.
        """
        if self._credentials is None:
            self._credentials, _ = google_auth_default(
                scopes=["https://www.googleapis.com/auth/cloud-healthcare"]
            )
        if not self._credentials.valid:
            self._credentials.refresh(GoogleAuthRequest())
        return self._credentials.token

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_access_token()}"}

    # ── Core HTTP with retries ────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        url: str,
        body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Authenticated HTTP request with exponential backoff retries.
        Retries on RETRYABLE_STATUS_CODES and connection errors.
        Raises FHIRClientError on non-retryable failures.
        """
        await self._init_session()
        headers = self._auth_headers()

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with self._session.request(
                    method, url,
                    headers=headers,
                    json=body,
                    params=params,
                ) as response:
                    raw = await response.text()

                    if response.status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                        wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                        logger.warning(
                            "FHIR %s %s → %s (attempt %d/%d). Retrying in %.1fs.",
                            method, url, response.status,
                            attempt + 1, MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        headers = self._auth_headers()
                        continue

                    if not response.ok:
                        raise FHIRClientError(
                            status=response.status,
                            method=method,
                            url=url,
                            body=raw,
                        )

                    return json.loads(raw) if raw else {}

            except aiohttp.ClientConnectionError as exc:
                if attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "FHIR connection error attempt %d/%d: %s. Retrying in %.1fs.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise FHIRClientError(
                        status=0, method=method, url=url, body=str(exc)
                    ) from exc

        raise FHIRClientError(status=0, method=method, url=url, body="Max retries exceeded")

    # ── FHIR CRUD ─────────────────────────────────────────────────────────────

    async def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        """
        Read a single FHIR resource by type and ID.

        Example:
            patient  = await client.read("Patient", "patient-123")
            coverage = await client.read("Coverage", "coverage-456")
        """
        url = f"{self._config.fhir_base_url}/{resource_type}/{resource_id}"
        logger.debug("FHIR READ %s/%s", resource_type, resource_id)
        try:
            return await self._request("GET", url)
        except FHIRClientError as exc:
            if exc.status == 404:
                raise FHIRNotFoundError(resource_type, resource_id) from exc
            raise

    async def search(
        self,
        resource_type: str,
        params: dict[str, str],
        max_pages: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for FHIR resources. Handles pagination up to max_pages.

        Example:
            conditions = await client.search(
                "Condition",
                {"patient": "patient-123", "clinical-status": "active"}
            )
        """
        url = f"{self._config.fhir_base_url}/{resource_type}"
        resources: list[dict[str, Any]] = []
        page = 0

        logger.debug("FHIR SEARCH %s params=%s", resource_type, params)

        while url and page < max_pages:
            bundle = await self._request(
                "GET", url, params=params if page == 0 else None
            )
            page += 1

            for entry in bundle.get("entry", []):
                if "resource" in entry:
                    resources.append(entry["resource"])

            url = next(
                (
                    link["url"]
                    for link in bundle.get("link", [])
                    if link.get("relation") == "next"
                ),
                None,
            )

        logger.debug("FHIR SEARCH %s → %d results", resource_type, len(resources))
        return resources

    async def create(
        self, resource_type: str, resource: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a new FHIR resource (POST). Server assigns the ID.

        Example:
            task = await client.create("Task", task_dict)
        """
        url = f"{self._config.fhir_base_url}/{resource_type}"
        logger.info("FHIR CREATE %s", resource_type)
        result = await self._request("POST", url, body=resource)
        logger.info("FHIR CREATE %s → ID: %s", resource_type, result.get("id", "?"))
        return result

    async def update(
        self,
        resource_type: str,
        resource_id: str,
        resource: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an existing FHIR resource (PUT).

        Example:
            updated = await client.update("Task", "task-123", updated_task_dict)
        """
        url = f"{self._config.fhir_base_url}/{resource_type}/{resource_id}"
        logger.info("FHIR UPDATE %s/%s", resource_type, resource_id)
        return await self._request("PUT", url, body=resource)

    async def patch(
        self,
        resource_type: str,
        resource_id: str,
        patch_body: list[dict],
    ) -> dict[str, Any]:
        """
        Partially update a FHIR resource using JSON Patch (RFC 6902).

        Example — promote a DocumentReference from preliminary to final:
            await client.patch("DocumentReference", "doc-123", [
                {"op": "replace", "path": "/docStatus", "value": "final"}
            ])
        """
        url = f"{self._config.fhir_base_url}/{resource_type}/{resource_id}"
        await self._init_session()
        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json-patch+json",
        }
        async with self._session.patch(url, headers=headers, json=patch_body) as resp:
            raw = await resp.text()
            if not resp.ok:
                raise FHIRClientError(
                    status=resp.status, method="PATCH", url=url, body=raw
                )
            return json.loads(raw)

    async def execute_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a FHIR transaction or batch bundle.
        Used by PA-4 to submit the PAS bundle as a single transaction.

        Example:
            result = await client.execute_bundle(pas_bundle)
        """
        url = self._config.fhir_base_url
        logger.info(
            "FHIR BUNDLE %s with %d entries",
            bundle.get("type", "?"),
            len(bundle.get("entry", [])),
        )
        return await self._request("POST", url, body=bundle)

    # ── Patient $everything ───────────────────────────────────────────────────

    async def everything(
        self,
        patient_id: str,
        resource_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Execute the FHIR $everything operation for a patient.
        Returns a Bundle of all resources linked to the patient.

        The PA Agent uses this to assemble clinical context for Gemini in PA-3.

        Example:
            bundle = await client.everything("patient-123", [
                "Condition", "Observation", "MedicationRequest",
                "AllergyIntolerance", "Coverage", "DiagnosticReport"
            ])
        """
        url = f"{self._config.fhir_base_url}/Patient/{patient_id}/$everything"
        params = {}
        if resource_types:
            params["_type"] = ",".join(resource_types)

        logger.info("FHIR $everything for patient %s", patient_id)
        bundle = await self._request("GET", url, params=params or None)

        count = len(bundle.get("entry", []))
        logger.info("FHIR $everything → %d resources for patient %s", count, patient_id)
        return bundle

    # ── Bundle helper ─────────────────────────────────────────────────────────

    @staticmethod
    def extract_resources(
        bundle: dict[str, Any],
        resource_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Extract resources from a FHIR Bundle, optionally filtered by type.

        Example:
            conditions = FHIRClient.extract_resources(bundle, "Condition")
        """
        resources = [
            entry["resource"]
            for entry in bundle.get("entry", [])
            if "resource" in entry
        ]
        if resource_type:
            resources = [r for r in resources if r.get("resourceType") == resource_type]
        return resources


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FHIRClientError(Exception):
    """Raised when the Healthcare API returns a non-2xx response."""

    def __init__(self, status: int, method: str, url: str, body: str):
        self.status = status
        self.method = method
        self.url = url
        self.body = body
        super().__init__(
            f"FHIR {method} {url} failed with status {status}: {body[:500]}"
        )


class FHIRNotFoundError(FHIRClientError):
    """Raised when a FHIR resource does not exist (HTTP 404)."""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(
            status=404,
            method="GET",
            url=f"{resource_type}/{resource_id}",
            body="Resource not found",
        )