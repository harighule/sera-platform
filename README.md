# SERA Intelligence Platform

SERA is a state-of-the-art behavioral intelligence platform designed to ingest and analyze multi-protocol telemetric event streams in real-time. It tracks, resolves, and evaluates high-dimensional entity states (across financial, healthcare, IoT, and social domains) using a robust mathematical foundation. By combining Shannon entropy analysis (AXIOM-Φ) for anomaly detection, custom PyTorch Continuous Interference Field Networks (CIFN) for causal inference, and real-time self-evolving predictive briefs (ZOLA) powered by Grok-3, SERA delivers predictive threat and opportunity briefs for enterprise environments.

## Quick Start (Docker)
1. Clone the repo to your local machine.
2. Copy `.env.example` to `.env` in the project root and fill in `AI_API_KEY` with your xAI Grok API key.
3. Run the services using Docker Compose:
   ```bash
   docker compose up --build
   ```
4. Open your browser and navigate to: http://localhost:3000
5. **Note**: All API requests require the following header for authorization:
   ```
   X-API-Key: sera-demo-2026
   ```
   *(or the value specified by your custom `DEMO_API_KEY` environment variable).*

## Quick Start (Local Dev)

### Backend
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open http://localhost:5173

## Architecture

| Layer | Component | Role |
|---|---|---|
| **SERA** | Data Ingestion | Multi-protocol event ingestion (SWIFT, HL7/FHIR, MQTT, HTTP) |
| **PRAGMA** | Semantic Manifold | Behavioral tensor space, entity resolution, consequence network |
| **AXIOM-Φ** | Entropy Engine | Shannon entropy analysis, pre-transition detection, z-score alerting |
| **The Entity** | CIFN / PyTorch | Continuous Interference Field Network, causal inference, self-evolution |
| **ZOLA** | Intelligence Layer | Behavioral predictions, intervention specs, reward settlement |

## Pages

| Page | URL | Description |
|---|---|---|
| **Dashboard** | `/` | Live telemetry stats, event stream, entropy chart |
| **Entity Explorer** | `/entities` | All 50 entities with entropy status and domain filters |
| **AXIOM Monitor** | `/axiom` | Entropy heatmap, alert feed, per-entity entropy inspector |
| **ZOLA Predictions** | `/zola` | Causal prediction briefs and KRONOS/CIFN optimization terminal |
| **AI Assistant** | `/ai` | Grok-3 powered chat with platform context |
| **Dark Intel** | `/intel` | Classified intelligence briefs with decrypt effects |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| **GET** | `/api/dashboard/stats` | Platform telemetry stats |
| **GET** | `/api/entities/` | List entities (supports limit and offset params) |
| **GET** | `/api/entities/{id}` | Get single entity by ID |
| **GET** | `/api/axiom/entropy` | Top entity entropy scores |
| **GET** | `/api/axiom/alerts` | Active pre-transition alerts |
| **GET** | `/api/zola/predictions` | Live causal prediction briefs |
| **GET** | `/api/zola/status` | KRONOS/CIFN model status |
| **POST** | `/api/zola/kronos/optimize` | Run one CIFN backprop step |
| **POST** | `/api/zola/evolve/propose` | Propose self-evolution patch |
| **POST** | `/api/zola/evolve/validate/{id}` | Sandbox validate patch |
| **POST** | `/api/zola/evolve/approve/{id}` | Apply patch to live model |
| **POST** | `/api/chat/` | Send message to Grok-3 AI assistant |
| **GET** | `/api/intel/news` | Live RSS news feed by domain |
| **GET** | `/api/intel/classified` | Classified intelligence briefs |
| **WS** | `/ws/stream` | Real-time event stream |

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| **AI_API_KEY** | Yes | — | xAI Grok-3 API key |
| **DEMO_API_KEY** | No | `your_demo_api_key_here` | API authentication key for all endpoints |
| **DATABASE_URL** | No | `postgresql+asyncpg://your_db_user:your_db_password@db:5432/sera_db` | Database connection string (e.g. Postgres or SQLite) |
| **ENTITY_MODE** | No | `live` | `live` uses PyTorch CIFN, `mock` uses random responses |

## Tech Stack
- **Backend**: FastAPI, SQLAlchemy async, PostgreSQL / SQLite, PyTorch, xAI Grok-3
- **Frontend**: React 18, Vite, Recharts, React Router v7
- **ML**: CIFN (Continuous Interference Field Network), LiveCausalNetwork, KRONOS orchestrator
- **Infrastructure**: Docker Compose, nginx, PostgreSQL 16

## Production Security & Deployment Requirements

> [!IMPORTANT]
> **Before deploying this service to any production or publicly accessible environment, you must adhere to the following safety guidelines:**
>
> 1. **TLS Termination / HTTPS**: This service does not configure TLS/HTTPS natively. It must run behind a TLS-terminating reverse proxy (e.g., Nginx, Caddy, or a cloud application load balancer) in any real deployment. Do not expose port 8000 directly to the internet without TLS termination.
> 2. **Credential Rotation**: Change all database usernames and passwords (`POSTGRES_USER` and `POSTGRES_PASSWORD` in `.env` / `docker-compose.yml`) from their default template values (`sera_user:sera_pass`) to secure, randomized credentials. Never check `.env` files into source control.
> 3. **API Key Authentication**: In production, do not use the single shared default API key (`sera-demo-2026`). Configure the `API_KEYS` environment variable with a JSON dictionary mapping secure, randomized keys to client identifiers (e.g. `{"secure-key-1": "client-a", "secure-key-2": "client-b"}`) to enforce per-client rate-limiting, revocation, and logging attribution.
> 4. **Rate Limiting**: Rate limiting is enforced globally at 60 requests/minute per client API key. Adjust this limit in `main.py` if client profiles require different quotas.

