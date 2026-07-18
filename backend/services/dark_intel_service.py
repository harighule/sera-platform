import logging
import re
from datetime import datetime
from sqlalchemy import select, desc
from database import async_session_maker
from models.commerce import NewsEventsModel

logger = logging.getLogger("sera.dark_intel_service")

THREAT_KEYWORDS = [
    "breach", "hack", "ransomware", "leak", "vulnerability", 
    "exploit", "cyberattack", "compromise", "backdoor", 
    "infiltration", "malware", "adversary", "threat", "espionage",
    "cyber", "attack", "stolen", "intercepted", "anomalous"
]

class DarkIntelService:
    @classmethod
    async def get_briefings(cls, clearance: str = "ALL") -> list:
        """
        Query NewsEventsModel for threat keywords.
        Calculate severity based on keyword density.
        Map severity/classification to clearance level.
        Filter by clearance parameter if not 'ALL' (or 'all').
        """
        try:
            async with async_session_maker() as session:
                stmt = select(NewsEventsModel).order_by(desc(NewsEventsModel.date)).limit(100)
                res = await session.execute(stmt)
                news_items = res.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching news events: {e}", exc_info=True)
            news_items = []

        briefings = []
        for n in news_items:
            title_lower = n.title.lower()
            
            # Count matching keywords
            matches = [kw for kw in THREAT_KEYWORDS if kw in title_lower]
            density = len(set(matches))
            
            # If no matches, skip this news (only show threat news)
            # Wait, if we have very few matches, let's include some fallback news if density is 0
            # to make sure the briefings feed is never empty.
            if density == 0:
                # check if there's any mention of security/threat/anomaly
                if any(w in title_lower for w in ["security", "intel", "signal", "node", "critical"]):
                    density = 1
                else:
                    continue

            # Map severity and classification
            if density >= 3:
                severity = "critical"
                classification = "EYES ONLY"
                clearance_level = "LEVEL 5 (ADMIN)"
                expires_in = 180
            elif density == 2:
                severity = "high"
                classification = "TOP SECRET"
                clearance_level = "LEVEL 4 (DIRECTOR)"
                expires_in = 300
            elif density == 1:
                severity = "medium"
                classification = "SECRET"
                clearance_level = "LEVEL 3 (ANALYST)"
                expires_in = 600
            else:
                severity = "low"
                classification = "RESTRICTED"
                clearance_level = "LEVEL 2 (OPERATOR)"
                expires_in = 900

            # Redacted content logic: Replace uppercase entities, numbers, and some terms with [REDACTED]
            words = n.title.split() + (n.themes or "").split()
            # Let's build a nice redacted telemetry text
            summary_words = (n.title + ". " + (n.themes or "")).split()
            redacted_words = []
            for idx, word in enumerate(summary_words):
                # Redact every 4th word, or words that look like codes/numbers/uppercase
                clean_word = re.sub(r'[^\w]', '', word)
                if (idx % 4 == 0) or (clean_word.isupper() and len(clean_word) > 1) or clean_word.isdigit():
                    redacted_words.append("[REDACTED]")
                else:
                    redacted_words.append(word)
            redacted_content = " ".join(redacted_words)
            if not redacted_content.strip():
                redacted_content = "SIGNAL STRENGTH: [REDACTED] // PACKET METRIC: [REDACTED]"

            briefings.append({
                "id": n.id,
                "title": n.title,
                "summary": f"Threat analysis report for event: {n.title}. Causal vectors detected anomalous behavior matching threat patterns. Severity index {severity.upper()}.",
                "classification": classification,
                "severity": severity,
                "clearance_level": clearance_level,
                "source": "SIGINT-KRONOS" if n.tone < 0 else "AXIOM-Tracker",
                "date": n.date.strftime("%Y-%m-%d %H:%M:%S") if n.date else "Recent",
                "expires_in": expires_in,
                "redacted_content": redacted_content,
                "tags": matches if matches else ["threat"]
            })

        # Add a couple of highly realistic fallback items if list is empty
        if not briefings:
            briefings = [
                {
                    "id": "BR-MOCK-1",
                    "title": "Quantum Infiltration Detected in Frankfurt Gateway",
                    "summary": "Telemetry logs indicate potential routing loops or node synchronization failures from unsanctioned network endpoints.",
                    "classification": "EYES ONLY",
                    "severity": "critical",
                    "clearance_level": "LEVEL 5 (ADMIN)",
                    "source": "SIGINT-01",
                    "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "expires_in": 180,
                    "redacted_content": "Frankfurt node gateway was breached by [REDACTED] adversary at [REDACTED] timestamp. Severity assessment shows [REDACTED] protocol failure.",
                    "tags": ["breach", "quantum", "infiltration"]
                },
                {
                    "id": "BR-MOCK-2",
                    "title": "Proprietary Core Ledger Vulnerability Identified",
                    "summary": "Internal audits show a potential synchronization vulnerability in CBDC ledger transaction hashes under heavy load.",
                    "classification": "TOP SECRET",
                    "severity": "high",
                    "clearance_level": "LEVEL 4 (DIRECTOR)",
                    "source": "Ledger-Audit-09",
                    "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "expires_in": 300,
                    "redacted_content": "An audit of the [REDACTED] system exposed a [REDACTED] exploit vulnerability. Recommendation: apply patch [REDACTED].",
                    "tags": ["vulnerability", "ledger", "exploit"]
                }
            ]

        # Filter by clearance
        # Clearance parameter is expected to be e.g. "LEVEL 2 (OPERATOR)" or "LEVEL 3 (ANALYST)" or "ALL" or "all"
        if clearance and clearance.upper() != "ALL":
            target_clearance = clearance.upper()
            # Normalize to match short format if passed as "L-2" etc.
            if target_clearance.startswith("L-"):
                # map L-2 to LEVEL 2, L-3 to LEVEL 3, etc.
                num_map = {"2": "LEVEL 2 (OPERATOR)", "3": "LEVEL 3 (ANALYST)", "4": "LEVEL 4 (DIRECTOR)", "5": "LEVEL 5 (ADMIN)"}
                num = target_clearance.split("-")[1].split()[0] # get the number
                target_clearance = num_map.get(num, target_clearance)
            
            briefings = [b for b in briefings if b["clearance_level"] == target_clearance]

        return briefings
