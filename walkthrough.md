# Walkthrough - Full Feature Set & Production Integrations

We have successfully built and verified the full feature set of the SERA platform, bridging the backend data processing, machine learning models, and complex database relationships with the premium React frontend pages.

---

## 1. APEX Causal Graph Viewer (Semantic Web UI)
- **Dynamic Node Expansions**: Fetches initial S&P 500 companies on page load. Clicking a node executes a GET request to `/api/semantic/outgoing/{ticker}` and dynamically grows neighbor nodes (Jobs, News, Shipping ports) from the center.
- **Color Coded Legends**: Neon styles distinguish node types (Cyan = Company, Gold = Job, Magenta = News, Green = Shipping).
- **Search & Zoom**: Supports camera centered animations when selecting search options.
- **Ingestion & Sync Sanitization**: Created a robust `sanitize_text` function to strip zero-width space characters (`\u200b`), mojibake (`â€‹`), and non-printable sequences during ingestion and database synchronization.
- **Ellipsis Suffix Mojibake & Encoding Repair**: Vite compilation encoding bugs were repaired by replacing Unicode ellipsis characters with standard ASCII sequences.

---

## 2. Claim Credibility (ALETHEIA) Engine
- **Database Schema**: Implemented `claims`, `evidence`, and `challenges` models in SQL Alchemy mapping to `sera_db.sqlite3`.
- **Scoring Algorithm**: Implemented a mathematical credibility score model calculated as:
  $$Credibility = \frac{BaseStake \cdot EvidenceWeight}{1 + ChallengePenalties}$$
  with support for reaffirmation and stakes adjustment.
- **REST Endpoints**: Created `/api/claims`, `/api/claims/{id}/challenge`, and `/api/claims/{id}/reaffirm` REST routing.
- **Frontend Integration**: Wired `ClaimCredibility.jsx` to dynamically submit stakes/claims, load exist claims by UUID, challenge claims, and display live score values with reactive animations.

---

## 3. AXIOM-Φ Monitor & Entropy Analytics
- **Entropy & Pre-Transition Calculations**: Computes Shannon entropy of entity news sentiment over 30-day and 90-day baselines, identifying entities exhibiting pre-transition volatility based on Z-score thresholding.
- **High-Risk Entities**: Exposes endpoints to search high-risk companies showing anomalously high informational entropy.
- **Frontend Integration**: Hooked `AxiomMonitor.jsx` into live endpoints to show real corporate entropy metrics, list pre-transition entities, and trigger manual node simulations.

---

## 4. ZOLA Causal Engine (Zola Predictions)
- **Predictions & Interventions**: Maps high-entropy transitions to recommended business decisions, optimal interventions, and consequence timelines.
- **Gödel Autogeneration Loop**: Simulates self-evolution patches targeting model weight fields to maximize generalization.
- **Live Predictor Integration**: Wired the frontend `ZolaPredictions.jsx` to parse target database fields (`legal_name`, `expansion_score`, and computed causal paths) dynamically, updating every 8 seconds.

---

## 5. Dark Intel & Classified Briefings
- **Threat Density Classification**: Automatically parses news reports for security threat keywords (breach, hack, leak, ransomware). Assigns severity level and classified clearances (EYES ONLY, TOP SECRET, SECRET, RESTRICTED).
- **Countdown De-authentication**: Integrates self-destruct timers. Cards lock and self-decrypt when clearance levels fail.
- **Clearance Selector**: Re-fetches the database dynamically when the clearance certificate changes.

---

## 6. Citation & Share of Voice (SOV) Tracking
- **SEO & GenAI Mentions**: Computes percentage share of voice and citation frequency of specific brand queries in news bodies over the last 30 days.
- **API endpoints**: Wired `CitationTracking.jsx` to write SEO queries to database via `/api/citation/track`, view history graphs, force citation recalculations via `/api/citation/run/{query_id}`, and fetch company-wide aggregate metrics via `/api/citation/rate`.
