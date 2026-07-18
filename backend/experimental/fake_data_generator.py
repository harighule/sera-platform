import random
import uuid
import numpy as np
from faker import Faker

fake = Faker()

# ----------------- FINANCIAL GENERATOR -----------------
CURRENCIES = ["USD", "eur", "GBP", "JPY", "CHF", "AUD", "CAD"]
REFERENCES = ["Invoice Payment", "Vendor Settlement","Monthly Payroll",
              "Intercompany Transfer", "Consulting Fees", "Reimbursement",
              "Dividend Payout", "Treasury Management"]

def generate_financial_event(entity_id, entity_name):
    amount = round(float(np.random.lognormal(mean=7.5, sigma=1.8)), 2)
    if amount < 1.0:
        amount = round(random.uniform(5.0, 50.0), 2)

    payload = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "sender_name": entity_name,
        "sender_account": fake.iban(),
        "sender_bank": fake.company(),
        "beneficiary_name": fake.name(),
        "beneficiary_account": fake.iban(),
        "beneficiary_bank": fake.company(),
        "amount": amount,
        "currency": random.choice(CURRENCIES),
        "reference": random.choice(REFERENCES),
        "swift_message_type": "MT103",
        "status": "COMPLETED"
    }

    return {"entity_id": entity_id, "name": entity_name, "protocol": "SWIFT",
            "event_type": "financial_transaction", "payload": payload}

# ----------------- HEALTHCARE GENERATOR -----------------
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

# ----------------- IOT GENERATOR -----------------
DEVICE_TYPES = ["temperature_sensor", "pressure_sensor", "motion_sensor",
                "humidity_sensor", "power_meter", "flow_sensor"]

def generate_iot_event(entity_id, entity_name):
    device_type = random.choice(DEVICE_TYPES)
    readings = {
        "temperature_sensor": {"temperature": round(random.normalvariate(22, 3), 2), "unit": "C"},
        "pressure_sensor": {"pressure": round(random.normalvariate(1013, 10), 2), "unit": "hPa"},
        "motion_sensor": {"motion_detected": random.random() < 0.2, "intensity": round(random.random(), 3)},
        "humidity_sensor": {"humidity": round(random.normalvariate(55, 10), 1), "unit": "%"},
        "power_meter": {"power_kw": round(abs(random.normalvariate(2.5, 1.2)), 3), "unit": "kW"},
        "flow_sensor": {"flow_rate": round(abs(random.normalvariate(15, 5)), 2), "unit": "L/min"},
    }
    payload = {
        "device_id": f"SENSOR-{uuid.uuid4().hex[:6].upper()}",
        "device_type": device_type,
        "entity_id": entity_id,
        "readings": readings[device_type],
        "battery": random.randint(10, 100),
        "signal_strength": random.randint(-90, -30),
        "location": {"lat": float(fake.latitude()), "lon": float(fake.longitude())},
        "firmware": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}"
    }
    return {"entity_id": entity_id, "name": entity_name, "protocol": "MQTT",
            "event_type": "iot_telemetry", "payload": payload}

# ----------------- SOCIAL GENERATOR -----------------
EVENT_TYPES = ["purchase", "search", "page_view", "login", "logout",
               "share", "comment", "profile_update", "subscription_change"]
PLATFORMS = ["web", "mobile_ios", "mobile_android", "api", "desktop"]
CATEGORIES = ["electronics", "food", "travel", "finance", "health",
              "entertainment", "education", "fashion", "home"]

def generate_social_event(entity_id, entity_name):
    event_type = random.choice(EVENT_TYPES)
    payload = {
        "event_type": event_type,
        "entity_id": entity_id,
        "platform": random.choice(PLATFORMS),
        "session_id": fake.uuid4(),
        "ip_address": fake.ipv4(),
        "user_agent": fake.user_agent(),
        "timestamp_ms": fake.unix_time() * 1000,
    }
    if event_type == "purchase":
        payload["amount"] = round(random.uniform(5, 2000), 2)
        payload["category"] = random.choice(CATEGORIES)
        payload["items"] = random.randint(1, 5)
    elif event_type == "search":
        payload["query"] = fake.sentence(nb_words=3)
        payload["results_count"] = random.randint(0, 500)
    return {"entity_id": entity_id, "name": entity_name, "protocol": "HTTP",
            "event_type": "behavioral_event", "payload": payload}


class FakeDataGenerator:
    """Consolidated class interface for generating fake behavioral and transactional signals."""
    
    @staticmethod
    def generate_random_event(entity_id: str, entity_name: str) -> dict:
        gen_fn = random.choice([
            generate_financial_event,
            generate_healthcare_event,
            generate_iot_event,
            generate_social_event
        ])
        return gen_fn(entity_id, entity_name)
