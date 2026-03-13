"""
scripts/load_synthetic_patient.py
Uses POST (server-assigned IDs) and saves logical->server ID map to data/test_ids.json
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import get_config
from shared.fhir_client import FHIRClient

PATIENT_ID       = "test-patient-thornton-001"
PRACTITIONER_ID  = "test-pract-chen-001"
ENCOUNTER_ID     = "test-enc-001"
COVERAGE_ID      = "test-coverage-bcbs-001"
CONDITION_DM_ID  = "test-cond-dm-001"
CONDITION_HTN_ID = "test-cond-htn-001"
CONDITION_CKD_ID = "test-cond-ckd-001"
OBS_HBA1C_ID     = "test-obs-hba1c-001"
OBS_GLUCOSE_ID   = "test-obs-glucose-001"
OBS_BP_ID        = "test-obs-bp-001"
OBS_BMI_ID       = "test-obs-bmi-001"
OBS_WEIGHT_ID    = "test-obs-weight-001"
MED_INSULIN_ID   = "test-med-insulin-001"
MED_METFORMIN_ID = "test-med-metformin-001"
ALLERGY_ID       = "test-allergy-pcn-001"
CI_ID            = "test-ci-001"
CAREPLAN_ID      = "test-cp-001"
DETECTED_ID      = "test-di-001"
CARETEAM_ID      = "test-careteam-001"

IDS_FILE = Path(__file__).parent.parent / "data" / "test_ids.json"

def load_id_map():
    if IDS_FILE.exists():
        return json.loads(IDS_FILE.read_text())
    return {}

def save_id_map(m):
    IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    IDS_FILE.write_text(json.dumps(m, indent=2))

async def delete_existing(fc):
    print("  Deleting existing test resources...")
    m = load_id_map()
    base = fc._config.fhir_base_url
    await fc._init_session()
    for lid, sid in reversed(list(m.items())):
        rt = lid.split("/")[0]
        url = base + "/" + rt + "/" + sid
        try:
            async with fc._session.delete(url, headers=fc._auth_headers()) as resp:
                print("    " + lid + " -> " + str(resp.status))
        except Exception:
            pass
    if IDS_FILE.exists():
        IDS_FILE.unlink()

async def load_resources(fc, delete_first=False):
    if delete_first:
        await delete_existing(fc)

    m = {}   # logical_id -> server_id
    errors = []

    def ref(lid):
        rt = lid.split("/")[0]
        sid = m.get(lid)
        return rt + "/" + sid if sid else lid

    def patient():
        return {"resourceType":"Patient",
            "identifier":[{"system":"urn:test:mrn","value":"MRN-TEST-001"}],
            "name":[{"use":"official","family":"Thornton","given":["James","Edward"]}],
            "gender":"male","birthDate":"1968-03-15",
            "address":[{"use":"home","line":["123 Synthetic Lane"],
                "city":"San Francisco","state":"CA","postalCode":"94102"}],
            "telecom":[{"system":"phone","value":"555-000-0000","use":"home"}]}

    def practitioner():
        return {"resourceType":"Practitioner",
            "identifier":[{"system":"http://hl7.org/fhir/sid/us-npi","value":"9999999999"}],
            "name":[{"use":"official","family":"Chen","given":["Sarah"]}],
            "qualification":[{"code":{"coding":[
                {"system":"http://terminology.hl7.org/CodeSystem/v2-0360","code":"MD"}]}}]}

    def care_team():
        return {"resourceType":"CareTeam","status":"active",
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "participant":[{"role":[{"coding":[{"system":"http://snomed.info/sct",
                "code":"59058001","display":"General physician"}]}],
                "member":{"reference":ref("Practitioner/"+PRACTITIONER_ID)}}]}

    def coverage():
        return {"resourceType":"Coverage","status":"active",
            "subscriber":{"reference":ref("Patient/"+PATIENT_ID)},
            "beneficiary":{"reference":ref("Patient/"+PATIENT_ID)},
            "payor":[{"identifier":{"system":"urn:test:payer-id","value":"bcbs-ca-001"},
                "display":"Blue Cross Blue Shield of California"}],
            "class":[{"type":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/coverage-class",
                "code":"plan"}]},"value":"PPO-GOLD-2026","name":"BCBS CA PPO Gold 2026"}],
            "period":{"start":"2026-01-01","end":"2026-12-31"}}

    def encounter():
        return {"resourceType":"Encounter","status":"finished",
            "class":{"system":"http://terminology.hl7.org/CodeSystem/v3-ActCode","code":"AMB"},
            "type":[{"coding":[{"system":"http://www.ama-assn.org/go/cpt","code":"99215"}],
                "text":"Office visit, established patient, high complexity"}],
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "participant":[{"individual":{"reference":ref("Practitioner/"+PRACTITIONER_ID)}}],
            "period":{"start":"2026-03-10T09:00:00Z","end":"2026-03-10T09:45:00Z"},
            "reasonCode":[{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm",
                "code":"E11.65","display":"Type 2 diabetes mellitus with hyperglycemia"}]}]}

    def condition_dm():
        return {"resourceType":"Condition",
            "clinicalStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-clinical","code":"active"}]},
            "verificationStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-ver-status","code":"confirmed"}]},
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-category","code":"problem-list-item"}]}],
            "code":{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm","code":"E11.65",
                "display":"Type 2 diabetes mellitus with hyperglycemia"}],"text":"Type 2 Diabetes Mellitus"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "encounter":{"reference":ref("Encounter/"+ENCOUNTER_ID)},
            "onsetDateTime":"2015-06-01","recordedDate":"2015-06-01",
            "asserter":{"reference":ref("Practitioner/"+PRACTITIONER_ID)}}

    def condition_htn():
        return {"resourceType":"Condition",
            "clinicalStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-clinical","code":"active"}]},
            "verificationStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-ver-status","code":"confirmed"}]},
            "code":{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm","code":"I10",
                "display":"Essential (primary) hypertension"}],"text":"Hypertension"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},"onsetDateTime":"2018-02-14"}

    def condition_ckd():
        return {"resourceType":"Condition",
            "clinicalStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-clinical","code":"active"}]},
            "verificationStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/condition-ver-status","code":"confirmed"}]},
            "code":{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm","code":"N18.3",
                "display":"Chronic kidney disease, stage 3 (moderate)"}],"text":"Chronic kidney disease stage 3"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},"onsetDateTime":"2022-09-01"}

    def obs_hba1c():
        return {"resourceType":"Observation","status":"final",
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"laboratory"}]}],
            "code":{"coding":[{"system":"http://loinc.org","code":"4548-4",
                "display":"Hemoglobin A1c/Hemoglobin.total in Blood"}],"text":"HbA1c"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "encounter":{"reference":ref("Encounter/"+ENCOUNTER_ID)},
            "effectiveDateTime":"2026-02-28","issued":"2026-02-28T14:30:00Z",
            "valueQuantity":{"value":8.2,"unit":"%","system":"http://unitsofmeasure.org","code":"%"},
            "referenceRange":[{"high":{"value":5.7,"unit":"%"},"text":"Normal: less than 5.7%"}],
            "interpretation":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code":"H","display":"High"}]}],
            "performer":[{"reference":ref("Practitioner/"+PRACTITIONER_ID)}]}

    def obs_glucose():
        return {"resourceType":"Observation","status":"final",
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"laboratory"}]}],
            "code":{"coding":[{"system":"http://loinc.org","code":"76629-5",
                "display":"Fasting glucose [Moles/volume] in Blood"}],"text":"Fasting glucose"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "effectiveDateTime":"2026-02-28",
            "valueQuantity":{"value":187,"unit":"mg/dL","system":"http://unitsofmeasure.org","code":"mg/dL"},
            "interpretation":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code":"H","display":"High"}]}]}

    def obs_bp():
        return {"resourceType":"Observation","status":"final",
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"vital-signs"}]}],
            "code":{"coding":[{"system":"http://loinc.org","code":"85354-9","display":"Blood pressure panel"}],
                "text":"Blood pressure"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "effectiveDateTime":"2026-03-10",
            "component":[
                {"code":{"coding":[{"system":"http://loinc.org","code":"8480-6","display":"Systolic BP"}]},
                 "valueQuantity":{"value":148,"unit":"mmHg"}},
                {"code":{"coding":[{"system":"http://loinc.org","code":"8462-4","display":"Diastolic BP"}]},
                 "valueQuantity":{"value":92,"unit":"mmHg"}}]}

    def obs_bmi():
        return {"resourceType":"Observation","status":"final",
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"vital-signs"}]}],
            "code":{"coding":[{"system":"http://loinc.org","code":"39156-5","display":"BMI"}]},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "effectiveDateTime":"2026-03-10",
            "valueQuantity":{"value":31.4,"unit":"kg/m2","system":"http://unitsofmeasure.org","code":"kg/m2"}}

    def obs_weight():
        return {"resourceType":"Observation","status":"final",
            "category":[{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/observation-category","code":"vital-signs"}]}],
            "code":{"coding":[{"system":"http://loinc.org","code":"29463-7","display":"Body weight"}]},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "effectiveDateTime":"2026-03-10",
            "valueQuantity":{"value":98.2,"unit":"kg","system":"http://unitsofmeasure.org","code":"kg"}}

    def med_insulin():
        return {"resourceType":"MedicationRequest","status":"active","intent":"order",
            "medicationCodeableConcept":{"coding":[{"system":"http://www.nlm.nih.gov/research/umls/rxnorm",
                "code":"274783","display":"insulin glargine 100 UNT/ML Injectable Solution"}],
                "text":"Insulin glargine (Lantus) 20 units subcutaneous at bedtime"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "encounter":{"reference":ref("Encounter/"+ENCOUNTER_ID)},
            "authoredOn":"2020-01-15",
            "requester":{"reference":ref("Practitioner/"+PRACTITIONER_ID)},
            "dosageInstruction":[{"text":"20 units subcutaneous injection at bedtime",
                "route":{"coding":[{"system":"http://snomed.info/sct","code":"34206005",
                    "display":"Subcutaneous route"}]},
                "doseAndRate":[{"doseQuantity":{"value":20,"unit":"units"}}]}],
            "reasonReference":[{"reference":ref("Condition/"+CONDITION_DM_ID)}]}

    def med_metformin():
        return {"resourceType":"MedicationRequest","status":"active","intent":"order",
            "medicationCodeableConcept":{"coding":[{"system":"http://www.nlm.nih.gov/research/umls/rxnorm",
                "code":"860975","display":"metformin HCl 1000 MG Oral Tablet"}],
                "text":"Metformin 1000mg twice daily with meals"},
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "authoredOn":"2015-06-15",
            "requester":{"reference":ref("Practitioner/"+PRACTITIONER_ID)},
            "dosageInstruction":[{"text":"1000mg orally twice daily with meals"}],
            "reasonReference":[{"reference":ref("Condition/"+CONDITION_DM_ID)}]}

    def allergy():
        return {"resourceType":"AllergyIntolerance",
            "clinicalStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical","code":"active"}]},
            "verificationStatus":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/allergyintolerance-verification","code":"confirmed"}]},
            "type":"allergy","category":["medication"],"criticality":"high",
            "code":{"coding":[{"system":"http://www.nlm.nih.gov/research/umls/rxnorm",
                "code":"7980","display":"Penicillin"}],"text":"Penicillin"},
            "patient":{"reference":ref("Patient/"+PATIENT_ID)},
            "onsetDateTime":"1995-01-01",
            "reaction":[{"manifestation":[{"coding":[{"system":"http://snomed.info/sct",
                "code":"39579001","display":"Anaphylaxis"}]}],"severity":"severe"}]}

    def clinical_impression():
        return {"resourceType":"ClinicalImpression","status":"completed",
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "encounter":{"reference":ref("Encounter/"+ENCOUNTER_ID)},
            "date":"2026-03-10T09:45:00Z",
            "assessor":{"reference":ref("Practitioner/"+PRACTITIONER_ID)},
            "description":("57-year-old male with T2DM (E11.65) on basal insulin (glargine 20u QHS) "
                "and metformin 1000mg BID. HbA1c 8.2% on 2026-02-28 indicating suboptimal glycemic "
                "control. Fasting glucose 187 mg/dL. BMI 31.4. HTN (I10), CKD stage 3 (N18.3). "
                "CGM clinically indicated for real-time glucose trending to optimize insulin dosing."),
            "finding":[
                {"itemCodeableConcept":{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm",
                    "code":"E11.65"}],"text":"T2DM with hyperglycemia"}},
                {"itemCodeableConcept":{"coding":[{"system":"http://hl7.org/fhir/sid/icd-10-cm",
                    "code":"Z79.4"}],"text":"Long-term insulin use"}}],
            "note":[{"text":"CGM authorization recommended. HbA1c above 7%, on insulin, T2DM confirmed."}]}

    def care_plan():
        return {"resourceType":"CarePlan","status":"active","intent":"proposal",
            "subject":{"reference":ref("Patient/"+PATIENT_ID)},
            "encounter":{"reference":ref("Encounter/"+ENCOUNTER_ID)},
            "period":{"start":"2026-03-10"},
            "title":"Diabetes Management Plan - CGM Initiation",
            "description":"Initiate CGM to optimize insulin titration for poorly controlled T2DM.",
            "addresses":[{"reference":ref("Condition/"+CONDITION_DM_ID)}],
            "activity":[{"detail":{
                "kind":"ServiceRequest",
                "code":{"coding":[{"system":"http://www.ama-assn.org/go/cpt","code":"95251",
                    "display":"Ambulatory continuous glucose monitoring"}],"text":"CGM - CPT 95251"},
                "status":"not-started",
                "description":"Initiate CGM device. 14-day wear. Physician review of trend data.",
                "performer":[{"reference":ref("Practitioner/"+PRACTITIONER_ID)}]}}],
            "careTeam":[{"reference":ref("CareTeam/"+CARETEAM_ID)}]}

    def detected_issue():
        return {"resourceType":"DetectedIssue","status":"final",
            "code":{"coding":[{"system":"http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code":"CLINRISK","display":"Clinical risk"}],"text":"Clinical context for CGM"},
            "severity":"moderate",
            "patient":{"reference":ref("Patient/"+PATIENT_ID)},
            "identifiedDateTime":"2026-03-10T09:45:00Z",
            "author":{"reference":ref("Practitioner/"+PRACTITIONER_ID)},
            "detail":"CKD stage 3 noted. No contraindication to CGM. No relevant drug interactions.",
            "implicated":[
                {"reference":ref("Condition/"+CONDITION_DM_ID)},
                {"reference":ref("Condition/"+CONDITION_CKD_ID)}]}

    ORDERED = [
        ("Patient/"+PATIENT_ID,              patient),
        ("Practitioner/"+PRACTITIONER_ID,    practitioner),
        ("CareTeam/"+CARETEAM_ID,            care_team),
        ("Coverage/"+COVERAGE_ID,            coverage),
        ("Encounter/"+ENCOUNTER_ID,          encounter),
        ("Condition/"+CONDITION_DM_ID,       condition_dm),
        ("Condition/"+CONDITION_HTN_ID,      condition_htn),
        ("Condition/"+CONDITION_CKD_ID,      condition_ckd),
        ("Observation/"+OBS_HBA1C_ID,        obs_hba1c),
        ("Observation/"+OBS_GLUCOSE_ID,      obs_glucose),
        ("Observation/"+OBS_BP_ID,           obs_bp),
        ("Observation/"+OBS_BMI_ID,          obs_bmi),
        ("Observation/"+OBS_WEIGHT_ID,       obs_weight),
        ("MedicationRequest/"+MED_INSULIN_ID,   med_insulin),
        ("MedicationRequest/"+MED_METFORMIN_ID, med_metformin),
        ("AllergyIntolerance/"+ALLERGY_ID,   allergy),
        ("ClinicalImpression/"+CI_ID,        clinical_impression),
        ("CarePlan/"+CAREPLAN_ID,            care_plan),
        ("DetectedIssue/"+DETECTED_ID,       detected_issue),
    ]

    for logical_id, builder in ORDERED:
        rt = logical_id.split("/")[0]
        resource = builder()
        try:
            result = await fc.create(rt, resource)
            server_id = result["id"]
            m[logical_id] = server_id
            print("  + " + logical_id + " -> " + server_id)
        except Exception as exc:
            msg = str(exc)[:200]
            errors.append(logical_id + ": " + msg)
            print("  X " + logical_id + " -- " + msg[:120])

    if m:
        save_id_map(m)
        print("")
        print("  ID map saved to: " + str(IDS_FILE))

    return m, errors


def save_questionnaire_template(config):
    template_dir = Path(__file__).parent.parent / "data" / "questionnaire_templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    payer_template = {
        "resourceType":"Questionnaire","id":"cgm-pa-bcbs-ca-001-95251",
        "url":"http://test.bcbsca.com/fhir/Questionnaire/cgm-pa","version":"2026.1",
        "name":"CGMPriorAuthorizationBCBSCA",
        "title":"Continuous Glucose Monitoring Prior Authorization - BCBS CA",
        "status":"active","subjectType":["Patient"],
        "item":[
            {"linkId":"q1","text":"Does the patient have a confirmed diagnosis of Type 1 or Type 2 Diabetes Mellitus?","type":"boolean","required":True},
            {"linkId":"q2","text":"What is the patient's most recent HbA1c value and date?","type":"string","required":True},
            {"linkId":"q3","text":"Is the patient currently on insulin therapy (basal, bolus, or pump)?","type":"boolean","required":True},
            {"linkId":"q4","text":"What insulin product and dose is the patient currently prescribed?","type":"string","required":True},
            {"linkId":"q5","text":"Does the patient have any contraindications to CGM device use?","type":"boolean","required":False},
            {"linkId":"q6","text":"What is the clinical justification for CGM at this time?","type":"string","required":True},
        ]}
    payer_path = template_dir / "bcbs-ca-001_95251.json"
    with open(payer_path, "w") as f:
        json.dump(payer_template, f, indent=2)
    print("  + Questionnaire template saved: " + str(payer_path))

    generic = {"resourceType":"Questionnaire","id":"generic-pa",
        "title":"Generic Prior Authorization Questionnaire","status":"active",
        "item":[
            {"linkId":"q1","text":"What is the primary diagnosis code (ICD-10) for this request?","type":"string","required":True},
            {"linkId":"q2","text":"Is this service medically necessary? Provide clinical justification.","type":"string","required":True},
            {"linkId":"q3","text":"Has the patient tried alternative treatments? If yes, describe.","type":"string","required":False},
            {"linkId":"q4","text":"What is the expected duration or frequency of this service?","type":"string","required":True},
        ]}
    generic_path = template_dir / "generic_pa.json"
    with open(generic_path, "w") as f:
        json.dump(generic, f, indent=2)
    print("  + Generic fallback template saved: " + str(generic_path))


def print_test_params(id_map):
    patient_id = id_map.get("Patient/"+PATIENT_ID, PATIENT_ID)
    encounter_id = id_map.get("Encounter/"+ENCOUNTER_ID, ENCOUNTER_ID)
    practitioner_id = id_map.get("Practitioner/"+PRACTITIONER_ID, PRACTITIONER_ID)
    print()
    print("=" * 60)
    print("  TEST PARAMETERS")
    print("=" * 60)
    print("  --patient-id      " + patient_id)
    print("  --cpt-code        95251")
    print("  --payer-id        bcbs-ca-001")
    print("  --encounter-id    " + encounter_id)
    print("  --practitioner-id " + practitioner_id)
    print()
    print("  Full agent CLI command:")
    print("  python -m agents.prior_auth.agent \\")
    print("    --patient-id=" + patient_id + " \\")
    print("    --cpt-code=95251 \\")
    print("    --payer-id=bcbs-ca-001 \\")
    print("    --encounter-id=" + encounter_id + " \\")
    print("    --practitioner-id=" + practitioner_id)
    print()
    print("  Expected: PA REQUIRED -> 6 HIGH confidence answers -> APPROVED")
    print("=" * 60)


async def main(delete_first=False, print_ids=False):
    config = get_config()
    print()
    print("=" * 60)
    print("  HC-CDSS PA Agent - Synthetic Patient Loader")
    print("  FHIR store: " + config.fhir_base_url)
    print("=" * 60)
    print()
    print("Step 1/3: Saving questionnaire templates...")
    save_questionnaire_template(config)
    print()
    print("Step 2/3: Loading FHIR resources...")
    async with FHIRClient(config) as fc:
        id_map, errors = await load_resources(fc, delete_first=delete_first)
    print()
    print("Step 3/3: Summary")
    print("  Loaded : " + str(len(id_map)) + " resources")
    print("  Errors : " + str(len(errors)))
    if errors:
        print()
        print("  ERRORS:")
        for e in errors:
            print("    " + e)
        sys.exit(1)
    if print_ids:
        print()
        print("  Logical ID -> Server ID:")
        for logical, server in id_map.items():
            print("    " + logical + " -> " + server)
    print_test_params(id_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-first", action="store_true")
    parser.add_argument("--print-ids", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(delete_first=args.delete_first, print_ids=args.print_ids))
