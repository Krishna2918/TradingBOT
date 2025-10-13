# Phase Blueprint

## Phase 1 – Repository scaffold & configuration baseline
- **Build**: Create the mandated `ai-trading-system/` structure, initialize configuration packages, and register environment loading so every module can source the verified credentials, model endpoints, and hardware assumptions.
- **Why now**: All later components rely on consistent paths, config contracts, and dependency management; establishing them first prevents rework.
- **Dependencies**: Consumes the existing `.env` artifacts and validated infrastructure; produces initialized config files (`agents.yaml`, `trading.yaml`, `risk.yaml`, `logging.yaml`) and package skeletons required downstream.
- **Inputs/Outputs**: Inputs—confirmed keys/models/hardware; Outputs—importable config modules, requirements manifest, logging setup used by every subsystem.

## Phase 2 – External service clients & market ingestion utilities
- **Build**: Implement authenticated clients for Alpha Vantage, NewsAPI, Finnhub, and Questrade, plus data acquisition scripts to populate the universe filters and historical datasets mandated for scoring.
- **Why here**: Data availability is prerequisite for feature engineering, AI scoring, and risk logic; implementing clients early exposes connectivity issues before upper layers depend on them.
- **Dependencies**: Requires Phase 1 configuration loading; outputs normalized API wrappers and raw dataset ingestion routines consumed by storage and feature layers.
- **Inputs/Outputs**: Inputs—service credentials; Outputs—structured market/news/fundamental payloads, throttling and error-handling utilities.

## Phase 3 – Persistence layer & data governance
- **Build**: Design schemas for `data/trading_state.db` and `data/market_data.duckdb`, along with repository services handling read/write pipelines for positions, history, indicators, sentiments, and AI scoring archives.
- **Why here**: Durable storage is required before higher-level modules can exchange information or power the dashboard.
- **Dependencies**: Consumes API outputs from Phase 2; delivers data access abstractions that agents, AI scorers, and dashboards will call.
- **Inputs/Outputs**: Inputs—raw ingested datasets; Outputs—database tables, ORM-like helpers, retention policies.

## Phase 4 – Feature engineering & multi-factor scoring framework
- **Build**: Implement computation pipelines for technical indicators, sentiment normalization, fundamental metrics, and momentum/volume analytics, mapping them into the weighted scoring schema.
- **Why here**: The AI ensemble depends on enriched features; constructing this layer after storage ensures reproducibility and caching.
- **Dependencies**: Needs persisted data from Phase 3; produces standardized feature matrices and scoring summaries for AI models and risk evaluators.
- **Inputs/Outputs**: Inputs—historical prices, news, fundamentals; Outputs—feature sets, normalized scores, metadata for AI prompts.

## Phase 5 – Ollama ensemble orchestration
- **Build**: Create interfaces for the three specified Ollama models, prompt templates, response validation, ensemble voting logic, and explanation synthesis adhering to the weighting strategy and daily selection requirement.
- **Why here**: Once features exist, the AI layer can transform them into actionable rankings; sequencing after Phase 4 ensures prompt payloads are ready.
- **Dependencies**: Requires feature outputs and config parameters; yields scored recommendations, confidence metrics, and rationale records consumed by agents and dashboards.
- **Inputs/Outputs**: Inputs—feature matrices, risk thresholds; Outputs—ranked shortlist, ensemble consensus metadata.

## Phase 6 – Agent orchestration & workflow controllers
- **Build**: Implement the eight agent classes covering discovery, scoring, risk evaluation, execution planning, monitoring, and logging, coordinating the daily timeline from pre-market preparation through market-close review.
- **Why here**: Agents glue together the AI, data, and trading services; they must encapsulate the chronological workflow once the underlying capabilities exist.
- **Dependencies**: Leverages AI outputs (Phase 5) and data services (Phases 2-4); emits orchestrated tasks, status events, and decision logs feeding risk and dashboard layers.
- **Inputs/Outputs**: Inputs—ensemble recommendations, market data, config schedules; Outputs—ordered task pipeline, staged orders, monitoring hooks.

## Phase 7 – Risk management & execution engine
- **Build**: Implement position sizing via Kelly criterion, exposure controls, circuit breakers, stop-loss automation, and Questrade order routing supporting paper trading, plus real-time P&L tracking.
- **Why here**: Execution and risk rely on agent coordination and AI signals; sequencing after Phase 6 ensures prerequisites exist.
- **Dependencies**: Requires orchestrated order intents and data stores; outputs validated orders, risk events, and execution logs for dashboards and persistence.
- **Inputs/Outputs**: Inputs—selected stocks, sizing constraints, market status; Outputs—order tickets, risk alerts, updated position states.

## Phase 8 – Dashboard integration
- **Build**: Incorporate the existing dashboard assets into `src/dashboard/`, rewire data sources to the new databases, and extend UI panels for AI selections, positions, rationales, performance, and risk monitoring without regressing existing views.
- **Why here**: Visualization depends on finalized data schemas and agent outputs; performing integration once upstream contracts stabilize prevents churn.
- **Dependencies**: Consumes data services, AI outputs, and risk/execution metrics; produces user interfaces and potentially APIs/websocket endpoints that other tooling might consume.
- **Inputs/Outputs**: Inputs—database connections, monitoring feeds; Outputs—interactive dashboards reflecting full system state.

## Phase 9 – Automation, testing, and readiness validation
- **Build**: Implement schedulers covering the 4 AM–market-close workflow, ensure environment variables drive behavior, and develop integration/performance tests meeting the mandated criteria before sign-off.
- **Why here**: After all features are in place, automation and verification ensure the system meets readiness and success metrics.
- **Dependencies**: Requires complete functional stack; outputs cron-compatible scripts, CI test suites, and validation reports demonstrating success criteria coverage.
- **Inputs/Outputs**: Inputs—full application components; Outputs—scheduled jobs, test harnesses, benchmark evidence.

## Build Execution Plan
1. Initialize the repository scaffold and configuration modules (Phase 1).
2. Stand up external API clients and ingestion routines, then persist raw data (Phases 2–3).
3. Layer feature engineering and the Ollama ensemble to produce daily stock rankings with explanations (Phases 4–5).
4. Implement agent-driven workflow control, followed by risk and execution mechanics to act on AI selections (Phases 6–7).
5. Integrate the legacy dashboard with the new data contracts and extend it with AI trading metrics (Phase 8).
6. Finish with scheduler automation, comprehensive testing, and readiness validation to satisfy production criteria (Phase 9).
