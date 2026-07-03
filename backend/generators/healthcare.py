import random, uuid
from faker import Faker

fake = Faker()
OBSERVATIONS = [
    {"code": "8867-4", "display": "Heart rate", "unit": "beats/min", "base_mean": 72, "base_std": 10, "anomalous_high": 145, "anomalous_low": 42},
    {"code": "15074-8", "display": "Blood Glucose", "unit": "mg/dL", "base_mean": 100, "base_std": 15, "anomalous_high": 280, "anomalous_low": 55},
    {"code": "8310-5", "display": "Body temperature", "unit": "Cel", "base_mean": 36.8, "base_std": 0.3, "anomalous_high": 39.5, "anomalous_low": 34.5},
    {"code": "2710-2", "display": "Oxygen saturation", "unit": "%", "base_mean": 98, "base_std": 1.5, "anomalous_high": 100, "anomalous_low": 85},
]

def generate_healthcare_event(entity_id, entity_name):
    obs = random.choice(OBSERVATIONS)
    is_anomaly = random.random() < 0.05

    if is_anomaly:
        value = float(obs["anomalous_high"] if random.random() < 0.5 else obs["anomalous_low"])
    else:
        value = round(float(random.normalvariate(obs["base_mean"], obs["base_std"])), 1)

    payload = {
        "resourceType": "Observation",
        "id": f"obs-{uuid.uuid4().hex[:12]}",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": obs["code"], "display": obs["display"]}]},
        "subject": {"reference": f"Patient/{entity_id}", "display": entity_name},
        "valueQuantity": {"value": value, "unit": obs["unit"]},
        "interpretation": "Abnormal" if is_anomaly else "Normal"
    }

    return {"entity_id": entity_id, "name": entity_name, "protocol": "FHIR",
            "event_type": "clinical_observation", "payload": payload}