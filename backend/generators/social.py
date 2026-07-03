import random
from faker import Faker

fake = Faker()
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