"""
langgraph_prior_auth/run.py

Two-phase runner demonstrating the human-in-the-loop interrupt.
Phase 1: PA-1 through PA-4 — pauses before payer submission.
Phase 2: Clinician reviews, then resumes PA-5.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from langgraph_prior_auth.graph import graph, PAState

THREAD_CONFIG = {"configurable": {"thread_id": "PA-DEMO-001"}}

INITIAL_STATE: PAState = {
    "patient_id": "db5fab6f-7e01-476d-a58f-efc7535529eb",
    "cpt_code": "95251",
    "payer_id": "bcbs-ca-001",
    "encounter_id": "e50a33e6-7398-459a-a11f-9112841ca955",
    "practitioner_id": "4efb9818-f9c8-4490-9b1e-815c95d7dd39",
    "pa_required": None,
    "crd_hook_card": None,
    "coverage_check_error": None,
    "questionnaire": None,
    "questionnaire_source": None,
    "dtr_fetch_error": None,
    "answers": None,
    "citation_failures": None,
    "citation_hard_fail": None,
    "filler_error": None,
    "pas_bundle": None,
    "dlp_findings": None,
    "dlp_blocked": None,
    "dlp_warnings": None,
    "assembler_error": None,
    "claim_response": None,
    "claim_response_id": None,
    "pubsub_message_id": None,
    "submit_error": None,
    "decision": None,
    "decision_reason": None,
}


def phase1():
    print("\n=== PHASE 1: PA-1 → PA-4 ===\n")

    for event in graph.stream(INITIAL_STATE, config=THREAD_CONFIG, stream_mode="updates"):
        node_name = list(event.keys())[0]
        updates = event[node_name]
        print(f"[{node_name}] completed")

        if node_name == "coverage_check":
            print(f"  PA required: {updates.get('pa_required')}")
        elif node_name == "dtr_fetch":
            print(f"  Questionnaire source: {updates.get('questionnaire_source')}")
        elif node_name == "questionnaire_filler":
            print(f"  Answers: {len(updates.get('answers') or [])}")
            print(f"  Citation failures: {len(updates.get('citation_failures') or [])}")
        elif node_name == "bundle_assembler":
            print(f"  DLP blocked: {updates.get('dlp_blocked')}")
            for w in (updates.get("dlp_warnings") or []):
                print(f"  WARNING: {w}")

    snapshot = graph.get_state(THREAD_CONFIG)
    interrupted = snapshot.next == ("pas_submit",)

    if not interrupted:
        state = snapshot.values
        print(f"\n=== TERMINATED EARLY ===")
        print(f"  Decision : {state.get('decision')}")
        print(f"  Reason   : {state.get('decision_reason')}")
        return False

    print("\n=== PAUSED — awaiting clinician review before submission ===")
    return True


def phase2():
    snapshot = graph.get_state(THREAD_CONFIG)
    state = snapshot.values

    print("\n--- Bundle summary ---")
    entries = (state.get("pas_bundle") or {}).get("entry", [])
    print(f"  Entries : {len(entries)}")

    print("\n--- DLP findings ---")
    findings = state.get("dlp_findings") or []
    if not findings:
        print("  None ✓")
    for f in findings:
        print(f"  {f.get('info_type')}: {f.get('likelihood')}")

    print("\n--- Citation failures ---")
    fails = state.get("citation_failures") or []
    if not fails:
        print("  None ✓")
    for f in fails:
        print(f"  linkId={f}")

    approved = input("\nApprove submission? (y/n): ").strip().lower() == "y"
    if not approved:
        print("Submission declined.")
        return

    print("\n=== PHASE 2: PA-5 submission ===\n")

    for event in graph.stream(None, config=THREAD_CONFIG, stream_mode="updates"):
        node_name = list(event.keys())[0]
        updates = event[node_name]
        print(f"[{node_name}] completed")
        print(f"  Decision        : {updates.get('decision')}")
        print(f"  ClaimResponse ID: {updates.get('claim_response_id')}")
        print(f"  Pub/Sub msg ID  : {updates.get('pubsub_message_id')}")

    final = graph.get_state(THREAD_CONFIG).values
    print(f"\n=== FINAL DECISION: {final.get('decision')} ===")
    print(f"  Reason: {final.get('decision_reason')}")


if __name__ == "__main__":
    interrupted = phase1()
    if interrupted:
        phase2()