import random, uuid
from faker import Faker

fake = Faker()
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