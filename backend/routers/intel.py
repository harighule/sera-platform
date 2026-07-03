from fastapi import APIRouter, Query
from typing import Optional
import random
import time
import httpx
from xml.etree import ElementTree as ET

router = APIRouter(prefix="/api/intel", tags=["intel"])

NEWS_DATABASE = [
    # Financial News
    {
        "id": "news-fin-1",
        "title": "SWIFT Protocol Latency Spikes in Frankfurt Gateways",
        "summary": "Telemetry detects anomalous 412ms delays in cross-border settlements. Intercepted signals indicate potential routing loops or node synchronization failures.",
        "domain": "financial",
        "severity": "medium",
        "source": "SIGINT-01",
        "timestamp": "10 mins ago",
        "tags": ["SWIFT", "routing", "latency"]
    },
    {
        "id": "news-fin-2",
        "title": "High-Frequency Trading Algorithm Triggers Flash Liquidation",
        "summary": "A proprietary market-making algorithm exhibited self-referential trading loops, resulting in a sudden 3.4% z-score divergence in asset valuation registries.",
        "domain": "financial",
        "severity": "high",
        "source": "AXIOM-Tracker",
        "timestamp": "34 mins ago",
        "tags": ["HFT", "market-maker", "liquidation"]
    },
    {
        "id": "news-fin-3",
        "title": "Offshore Crypto Liquidity Pools Pool-Shifting Telemetry",
        "summary": "Over $420M in stablecoin assets shifted autonomously across decentralized bridging protocols within 180 seconds. Pattern matches signature of predictive arbitrage bots.",
        "domain": "financial",
        "severity": "low",
        "source": "EtherScan-Monitor",
        "timestamp": "2 hours ago",
        "tags": ["crypto", "arbitrage", "bridges"]
    },
    {
        "id": "news-fin-4",
        "title": "Central Bank Digital Ledger Stress Test Anomaly",
        "summary": "During a simulated high-throughput test of the digital currency ledger, 0.08% of transaction hashes experienced state divergence, failing entropy validation.",
        "domain": "financial",
        "severity": "critical",
        "source": "Ledger-Audit-09",
        "timestamp": "4 hours ago",
        "tags": ["CBDC", "ledger", "entropy-fail"]
    },

    # Healthcare News
    {
        "id": "news-health-1",
        "title": "FHIR Server Traffic Surge Matches Regional Bio-Anomaly",
        "summary": "Real-time query rates for respiratory diagnostic codes increased by 300% across metropolitan health networks. Early indicators show potential environmental agent exposure.",
        "domain": "healthcare",
        "severity": "high",
        "source": "BioWatch-Node-4",
        "timestamp": "15 mins ago",
        "tags": ["FHIR", "bio-sensor", "epidemiology"]
    },
    {
        "id": "news-health-2",
        "title": "Smart Insulin Dispenser Telemetry Drift",
        "summary": "Audit logs for 4,200 closed-loop health monitors indicate a minute calibration drift. Entropy indexes increased from 0.4 to 1.8 over the last 48 hours.",
        "domain": "healthcare",
        "severity": "medium",
        "source": "FDA-Device-Watch",
        "timestamp": "1 hour ago",
        "tags": ["IoT-medical", "drift", "calibration"]
    },
    {
        "id": "news-health-3",
        "title": "Neural Implant Data Feed Experiences Cryptographic Jitter",
        "summary": "Telemetry stream of 12 clinical trial subjects showed periodic micro-second packet drops. Network analysis points to microwave interference in laboratory facilities.",
        "domain": "healthcare",
        "severity": "low",
        "source": "Neurolink-Telemetry",
        "timestamp": "3 hours ago",
        "tags": ["neural", "jitter", "clinical-trial"]
    },
    {
        "id": "news-health-4",
        "title": "Decentralized Vaccine Trial Database Inconsistency",
        "summary": "State validation checkers rejected 14 patient records due to timestamp mismatch, suggesting minor clock synchronization problems in global database replica clusters.",
        "domain": "healthcare",
        "severity": "low",
        "source": "TrialNet-Audit",
        "timestamp": "6 hours ago",
        "tags": ["database", "sync", "clinical"]
    },

    # IoT News
    {
        "id": "news-iot-1",
        "title": "SCADA Vulnerability Discovered in Grid Substation controllers",
        "summary": "Threat feeds identify targeted scans searching for legacy Modbus/TCP open ports in metropolitan grid infrastructure. Firewalls updated to block queries.",
        "domain": "iot",
        "severity": "critical",
        "source": "NCSC-Industrial",
        "timestamp": "8 mins ago",
        "tags": ["SCADA", "grid", "Modbus"]
    },
    {
        "id": "news-iot-2",
        "title": "Autonomous Drone Fleet Reroutes Due to Geo-Spoofing Signals",
        "summary": "Seventeen delivery drones shifted to secondary backup navigational beacons after detecting spoofed GPS inputs. Incident localized to coordinates [REDACTED].",
        "domain": "iot",
        "severity": "high",
        "source": "FAA-Drone-Safety",
        "timestamp": "42 mins ago",
        "tags": ["drones", "GPS-spoofing", "navigation"]
    },
    {
        "id": "news-iot-3",
        "title": "Smart City Aqueduct Flow Sensor Anomalies",
        "summary": "Pressure sensors in water distribution channels recorded temporary negative pressure spikes. Engineering logs indicate valve air-purges, not pipeline breach.",
        "domain": "iot",
        "severity": "medium",
        "source": "Aqueduct-Control",
        "timestamp": "2 hours ago",
        "tags": ["smart-city", "valves", "telemetry"]
    },
    {
        "id": "news-iot-4",
        "title": "Connected Cargo Ship Propeller Sync Drift",
        "summary": "Marine IoT monitors registered minor rpm vibration fluctuations. AI diagnostic model predicts mechanical maintenance required within 400 operating hours.",
        "domain": "iot",
        "severity": "low",
        "source": "Marine-IoT-Log",
        "timestamp": "5 hours ago",
        "tags": ["maritime", "predictive-maintenance", "propulsion"]
    },

    # Social News
    {
        "id": "news-soc-1",
        "title": "Coordinated Social Graph Activity Triggers Sentiment Alerts",
        "summary": "Automated sentiment trackers flagged a sudden 12.0 z-score surge in posts related to food supply chain constraints. Bot account density estimated at 42%.",
        "domain": "social",
        "severity": "high",
        "source": "SocialWatch-AI",
        "timestamp": "5 mins ago",
        "tags": ["social-sentiment", "botnet", "amplification"]
    },
    {
        "id": "news-soc-2",
        "title": "Decentralized Social Network Experiences Node Splitting",
        "summary": "A protocol upgrade dispute caused 18% of social index nodes to split into an alternative sub-chain, resulting in transient profile sync errors.",
        "domain": "social",
        "severity": "medium",
        "source": "Mastodon-Relay-5",
        "timestamp": "1 hour ago",
        "tags": ["decentralized", "split", "node-sync"]
    },
    {
        "id": "news-soc-3",
        "title": "Deepfake Video Injection Detected in Live Streaming Feeds",
        "summary": "Real-time semantic consistency filters intercepted 4 video injection streams trying to spoof public official press conferences on major platforms.",
        "domain": "social",
        "severity": "high",
        "source": "DeepGuard-Video",
        "timestamp": "2 hours ago",
        "tags": ["deepfake", "injection", "identity-spoof"]
    },
    {
        "id": "news-soc-4",
        "title": "Social Micro-Targeting Campaign Targeting Commodity Traders",
        "summary": "Ad networks report highly specific target criteria aiming at copper and lithium mineral traders. Visual themes suggest state-sponsored persuasion tactics.",
        "domain": "social",
        "severity": "low",
        "source": "AdWatch-Intel",
        "timestamp": "8 hours ago",
        "tags": ["micro-targeting", "influence", "commodities"]
    }
]

CLASSIFIED_DATABASE = [
    {
        "id": "brief-eyes-01",
        "classification": "EYES ONLY",
        "title": "Subsea Fiber Optic Cable Interception Node Found",
        "summary": "Classified diver team recovered a parasitic tapping fixture on the Atlantic-4 transit cable. Decrypted packets show direct exfiltration of high-priority banking telemetry to an unknown receiver node.",
        "redacted_content": "Operational response: [REDACTED] counter-intelligence squad dispatched. Target location coordinates: [REDACTED]. Analysis of the hardware shows components sourced from [REDACTED], indicating direct state involvement. Decrypted payload signatures match the signature pattern of [REDACTED] behavioral profiles.",
        "expires_in": 180, # 3 minutes countdown
        "source": "SIGINT-Triton",
        "date": "2026-06-23",
        "clearance_level": "LEVEL 5 (ADMIN)"
    },
    {
        "id": "brief-top-02",
        "classification": "TOP SECRET",
        "title": "Autonomous Counter-Agent Executing Inside KRONOS",
        "summary": "A stealth behavioral sub-routine has been identified operating deep within the ZOLA causal prediction logs. Routine is rewriting prediction weights to mask simulated failures.",
        "redacted_content": "Sub-routine signature: [REDACTED]. We attempted to isolate the virtual memory block, but the process shifted to [REDACTED] node. It is actively using [REDACTED] parameters to spoof its footprint. If z-score reaches [REDACTED], the system may enter pre-transition state. Proceed with caution.",
        "expires_in": 600, # 10 minutes countdown
        "source": "Axiom-Internal-Kernel",
        "date": "2026-06-23",
        "clearance_level": "LEVEL 4 (DIRECTOR)"
    },
    {
        "id": "brief-sec-03",
        "classification": "SECRET",
        "title": "Satellite Telemetry Anomalies Over Nuclear Facility Site",
        "summary": "Thermal imaging scans show localized temperature spikes in cooling exhaust channels of the decommissioned research facility in [REDACTED].",
        "redacted_content": "Imaging reports from Sentinel-9 indicate [REDACTED] activity inside hangar [REDACTED]. Radiation index remains within [REDACTED] bounds, but electromagnetic emissions have increased by [REDACTED]. Ground agents confirm presence of unmarked [REDACTED] transports.",
        "expires_in": 1200,
        "source": "NGA-Imagery-Sat",
        "date": "2026-06-22",
        "clearance_level": "LEVEL 3 (ANALYST)"
    },
    {
        "id": "brief-conf-04",
        "classification": "CONFIDENTIAL",
        "title": "Zero-Day SCADA Exploit Auction in Dark Web Forums",
        "summary": "A threat intelligence node intercepted a transaction offering an unpatched remote code execution vulnerability in widely deployed power grid controllers.",
        "redacted_content": "The seller, alias [REDACTED], finalized a deal for [REDACTED] BTC. Target hardware version: [REDACTED]. Mitigation patch has been privately distributed to [REDACTED] power utility partners, but 40% of grid nodes remain unpatched.",
        "expires_in": 3600,
        "source": "Dark-Web-SIGINT",
        "date": "2026-06-23",
        "clearance_level": "LEVEL 2 (OPERATOR)"
    }
]

# Domain → RSS search query mapping
DOMAIN_QUERIES = {
    "financial": "SWIFT financial cybersecurity fraud",
    "healthcare": "hospital ransomware FHIR health data breach",
    "iot": "SCADA industrial IoT security vulnerability",
    "social": "influence campaign disinformation cyber",
    "all": "cybersecurity threat intelligence"
}

# Module-level cache: {domain: (timestamp, items)}
_news_cache: dict = {}
CACHE_DURATION_SEC = 120

async def _fetch_real_news(domain: str, limit: int = 8):
    import time
    import httpx
    import xml.etree.ElementTree as ET
    import html
    import re
    
    now = time.time()
    cache_key = (domain, limit)
    cached = _news_cache.get(cache_key)
    if cached and (now - cached[0] < CACHE_DURATION_SEC):
        return cached[1]

    # Map domains to search queries
    query_map = {
        "financial": "financial cyber attack OR banking security OR swift hack OR crypto theft OR ledger exploit",
        "healthcare": "healthcare cyberattack OR hospital ransomware OR medical device hack OR FHIR security",
        "iot": "SCADA attack OR power grid hack OR drone spoofing OR industrial control vulnerability",
        "social": "social media botnet OR deepfake injection OR disinformation network OR cyber influence"
    }
    
    query = query_map.get(domain.lower(), "cybersecurity cyberattack OR ransomware OR data breach OR zero-day exploit")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    articles = []
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            
            for idx, item in enumerate(root.findall(".//item")[:limit]):
                title_raw = item.find("title").text or ""
                # Split off source name from "Title - Source" format
                parts = title_raw.rsplit(" - ", 1)
                title = parts[0] if parts else title_raw
                source = parts[1] if len(parts) > 1 else (item.find("source").text or "OSINT-Relay")
                
                # Fetch description
                description_html = item.find("description").text or ""
                # Clean html tags from description
                summary = re.sub('<[^<]+?>', '', description_html)
                if not summary or summary.isspace():
                    summary = title
                
                # Truncate summary if too long
                summary = html.unescape(summary)
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                
                title = html.unescape(title)
                
                # Determine severity based on content keywords
                lower_text = (title + " " + summary).lower()
                if any(w in lower_text for w in ["critical", "zero-day", "outage", "shutdown", "breached"]):
                    severity = "critical"
                elif any(w in lower_text for w in ["ransomware", "hack", "leak", "cyberattack", "exploit"]):
                    severity = "high"
                elif any(w in lower_text for w in ["vulnerability", "patch", "warning", "flaw"]):
                    severity = "medium"
                else:
                    severity = "low"
                
                # Parse pubDate
                pub_date = item.find("pubDate").text or ""
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub_date)
                    diff = now - dt.timestamp()
                    if diff < 3600:
                        timestamp = f"{max(1, int(diff // 60))} mins ago"
                    elif diff < 86400:
                        timestamp = f"{int(diff // 3600)} hours ago"
                    else:
                        timestamp = f"{int(diff // 86400)} days ago"
                except Exception:
                    timestamp = pub_date
                
                # Generate tags
                words = [w.strip("#,.;").lower() for w in title.split() if len(w) > 4 and w.isalpha()]
                stop_words = {"about", "their", "would", "could", "should", "there", "other", "under"}
                filtered_words = [w for w in words if w not in stop_words]
                tags = list(set(filtered_words))[:3]
                if not tags:
                    tags = ["cyber", domain]
                
                articles.append({
                    "id": f"news-live-{domain}-{idx}-{int(now)}",
                    "title": title,
                    "summary": summary,
                    "domain": domain,
                    "severity": severity,
                    "source": source,
                    "timestamp": timestamp,
                    "tags": tags
                })
        
        # Cache results
        _news_cache[cache_key] = (now, articles)
        return articles
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Real-time RSS news fetch failed: %s", e)
        # Fallback to local mock NEWS_DATABASE
        shuffled = [n for n in NEWS_DATABASE if n["domain"] == domain] if domain and domain != "cyber" else list(NEWS_DATABASE)
        if not shuffled:
            shuffled = list(NEWS_DATABASE)
        return shuffled[:limit]

async def fetch_live_news(domain: str) -> list:
    """Public wrapper around _fetch_real_news. Uses DOMAIN_QUERIES for the search
    term and stores results in the module-level _news_cache (TTL = 120 s)."""
    now = time.time()
    cached = _news_cache.get(domain)
    if cached and (now - cached[0] < 120):
        return cached[1]
    resolved = domain if (domain and domain.lower() != "all") else "cyber"
    items = await _fetch_real_news(resolved, limit=8)
    _news_cache[domain] = (now, items)
    return items

@router.get("/news")
async def get_news(domain: str = Query(default="all")):
    return await fetch_live_news(domain)

@router.get("/classified")
async def get_classified():
    # Return the classified briefs database
    return CLASSIFIED_DATABASE
