# LangGraph Prior Authorization Agent

> **Framework comparison:** This is a LangGraph rebuild of the [ADK Prior Authorization Agent](https://github.com/gbhorne/adk-prior-authorization-agent). Same clinical pipeline, same GCP stack, different orchestration architecture. Built to demonstrate the tradeoffs between LLM-driven orchestration (Google ADK) and deterministic graph execution (LangGraph).

A production-grade Prior Authorization Agent that automates the full PA lifecycle — from coverage verification through payer submission — using the Da Vinci CRD/DTR/PAS trilogy, Google Cloud Healthcare FHIR R4, and LangGraph's deterministic StateGraph with native human-in-the-loop interrupts.

---

## Why Two Frameworks?

The [ADK version](https://github.com/gbhorne/adk-prior-authorization-agent) lets Gemini 2.5 Flash orchestrate the pipeline autonomously — deciding which tool to call next, handling ambiguous questionnaire answers, and surfacing early-exit conditions without a single explicit branch. It completed the full PA-1 through PA-5 pipeline in ~81 seconds end-to-end.

This LangGraph version rebuilds the same pipeline with deterministic graph edges. Every routing decision is explicit code. Every state transition is typed. The human review gate is a native graph interrupt — not an upstream workflow convention.

| Dimension | ADK | LangGraph |
|---|---|---|
| Orchestration | Gemini reasons over tool list | Deterministic graph edges |
| State | Tool return values in LLM context | TypedDict, persisted per node |
| Early exits | LLM exception handling | Conditional edge functions |
| Human review | ClaimResponse written as `draft` | `interrupt_before=["pas_submit"]` |
| Observability | ADK Web UI trace panel | LangSmith + LangGraph Studio |
| Best for | Ambiguous clinical Q&A | Compliance-auditable pipelines |

Neither is better. The choice depends on the requirement. This build demonstrates the judgment to match architecture to use case.

---

## What It Does

A clinician orders CPT 95251 (Continuous Glucose Monitoring) for a patient with Type 2 Diabetes. The agent autonomously:

1. **PA-1** — Calls the payer's CDS Hooks CRD endpoint to determine if PA is required
2. **PA-2** — Fetches the payer's DTR questionnaire (with Firestore cache)
3. **PA-3** — Uses Gemini to answer each questionnaire item with citations to real FHIR resources
4. **PA-4** — Assembles a Da Vinci PAS-compliant FHIR transaction bundle and runs Cloud DLP audit
5. **⏸ INTERRUPT** — Graph pauses. Clinician inspects bundle, DLP findings, and citation failures
6. **PA-5** — On clinician approval, submits to payer `$submit`, writes ClaimResponse to FHIR, publishes to Pub/Sub

---

## Architecture

```
__start__
    │
    ▼
coverage_check ──[PA not required or error]──► __end__
    │
    ▼ [PA required]
dtr_fetch
    │
    ▼
questionnaire_filler ──[citation hard fail or error]──► __end__
    │
    ▼ [answers complete]
bundle_assembler ──[DLP block or error]──► __end__
    │
    ▼ [DLP passed]
⏸ INTERRUPT — clinician review
    │
    ▼ [approved]
pas_submit
    │
    ▼
__end__
```

### Human-in-the-loop

The graph compiles with `interrupt_before=["pas_submit"]`. After PA-4 completes, the graph freezes — no LLM running, no GCP cost accruing. The clinician retrieves state, inspects the assembled PAS bundle and DLP findings, then resumes with `graph.invoke(None, config=config)`. Every state snapshot is stored in the checkpointer with a timestamp — full audit trail per `thread_id`.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.1+ |
| LLM | Gemini 2.5 Flash |
| FHIR | Google Cloud Healthcare API R4 |
| PA standard | Da Vinci CRD + DTR + PAS (HL7 FHIR) |
| Compliance | CMS-0057-F (payer interoperability rule) |
| PHI audit | Cloud DLP |
| State / cache | Firestore |
| Messaging | Cloud Pub/Sub |
| Auth secrets | Secret Manager |
| Observability | LangSmith + LangGraph Studio |
| Runtime | Python 3.11 · Cloud Run (prod) |

---

## Project Structure

```
langgraph-prior-authorization-agent/
├── langgraph_prior_auth/
│   ├── __init__.py
│   ├── graph.py          # StateGraph — 5 nodes, conditional edges, interrupt
│   └── run.py            # Two-phase runner with human-in-the-loop demo
├── agents/prior_auth/
│   ├── agent.py          # Original async PA orchestrator
│   └── tools/
│       ├── coverage_check.py       # PA-1: CDS Hooks CRD
│       ├── dtr_fetch.py            # PA-2: DTR questionnaire
│       ├── questionnaire_filler.py # PA-3: Gemini Q&A
│       ├── bundle_assembler.py     # PA-4: PAS bundle + DLP
│       └── pas_submit.py           # PA-5: $submit + FHIR write
├── shared/
│   ├── config.py         # CDSSConfig + Secret Manager
│   ├── fhir_client.py    # Google Cloud Healthcare FHIR client
│   └── models.py         # Pydantic models
├── scripts/
│   ├── mock_payer_server.py        # Local CRD/DTR/PAS mock (aiohttp)
│   └── load_synthetic_patient.py  # FHIR test data loader
├── langgraph.json        # LangGraph dev server config
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/gbhorne/langgraph-prior-authorization-agent
cd langgraph-prior-authorization-agent
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows
pip install -r requirements.txt
```

Configure `.env`:

```
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
FHIR_DATASET=cdss-dataset
FHIR_DATASTORE=cdss-fhir-store
FHIR_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
PAYER_ENDPOINTS={"bcbs-ca-001": "http://localhost:8080/crd"}
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=prior-auth-langgraph
```

### Run with the two-phase runner

```powershell
# Terminal 1 — mock payer server
python scripts/mock_payer_server.py

# Terminal 2 — run the graph
python -m langgraph_prior_auth.run
```

Phase 1 runs PA-1 through PA-4 and pauses. Review the bundle summary and DLP findings, then type `y` to approve and fire PA-5.

### Run with LangGraph Studio

```powershell
# Terminal 1 — mock payer server
python scripts/mock_payer_server.py

# Terminal 2 — LangGraph dev server
langgraph dev
```

Open the Studio URL printed in the terminal. Enter patient parameters in the input form and click Submit to watch nodes execute in real time.

---

## LangSmith Traces

Every run is traced to LangSmith with per-node latency, state diffs, and token counts.

| Node | Typical latency |
|---|---|
| coverage_check | 1.8s |
| dtr_fetch | 0.1s (cache hit) / 1.6s (miss) |
| questionnaire_filler | 6–20s (Gemini) |
| bundle_assembler | 1.5s |
| pas_submit | 0.8s |

---

## GCP Infrastructure

| Resource | Details |
|---|---|
| Healthcare API dataset | `cdss-dataset` (us-central1) |
| FHIR R4 store | `cdss-fhir-store` |
| Firestore | `(default)` native mode |
| Pub/Sub topics | `orchestrator-ready`, `prior-auth-ready` |
| Secrets | `pa-payer-endpoints`, `availity-client-id`, `availity-client-secret` |

---

## Disclaimer

All patient data, payer identifiers, clinical records, and submissions in this repository are entirely fictitious and generated for demonstration purposes only. No real patient information is used. Not intended for production use without further hardening, security review, and compliance validation.

---

## Author

**Gregory Horne** · [github.com/gbhorne](https://github.com/gbhorne)

---

## Related

- [ADK Prior Authorization Agent](https://github.com/gbhorne/adk-prior-authorization-agent) — the original LLM-orchestrated version
- [ADK Retail Agents](https://github.com/gbhorne/adk-retail-agents) — multi-agent retail analytics on Google ADK
