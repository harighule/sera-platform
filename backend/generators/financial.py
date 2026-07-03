import random, uuid
from faker import Faker
import numpy as np

fake = Faker()
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