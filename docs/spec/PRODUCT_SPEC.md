# Mezvuat AI + ModelMesh — Product Vision, Functional Specification and Integration Architecture

> Source of truth written by the product owner. Both repositories carry an identical copy of this file.
> Each agent owns only its own product; the other product's section is here for integration context.
> Note: the responsibility table in §4.2 was reconstructed from the prose of the spec (the original checkmarks were lost in transfer).

There are two independent products, each with its own purpose, architecture, responsibilities, APIs and tests. They are not two modules of one application.

- **Mezvuat AI** is the intelligence product.
- **ModelMesh** is the inference infrastructure product.

Mezvuat must be usable without ModelMesh, through direct provider integrations. ModelMesh must be usable independently by any compatible application. Their integration should demonstrate why the products are better together.

The objective is a portfolio that demonstrates AI application engineering, applied machine learning, model evaluation, inference optimization and production-oriented software engineering.

---

# 1. Product One: Mezvuat AI

*Research intelligence. Connected knowledge. Verifiable answers.*

## 1.1 Product vision

Mezvuat AI is an AI-powered research and knowledge intelligence platform that helps users investigate complex subjects using multiple sources of information. It goes beyond conventional RAG chatbots (embed a document, retrieve a few chunks, ask an LLM).

Mezvuat should understand the research task, retrieve evidence intelligently, determine what additional information is needed, reason across sources and produce answers that can be traced back to their supporting evidence.

**Product promise:** every meaningful answer is grounded in evidence, every important claim is traceable, and the system can evaluate the quality of its own output.

## 1.2 Who is it for?

Researchers, students doing literature reviews, analysts comparing documents and reports, professionals working with large technical documentation collections, anyone synthesizing information from multiple sources.

The initial product supports a focused, manageable document collection, not internet crawling. Candidate demo domains: technical research papers, regulatory documents, public institutional reports. The architecture must remain domain-independent.

## 1.3 Core capabilities

### A. Knowledge ingestion and management

Users create research workspaces and populate them with source material.

Initial inputs: PDF; TXT and Markdown; public document URLs where extraction is supported.

Every ingested document has a persistent identity and metadata: title and source; publication date; version (where available); ingestion timestamp; content hash; author / issuing organization (where available); page numbers and section information; document relationships.

Ingestion pipeline:
1. Validate incoming files.
2. Extract text and metadata.
3. Preserve document structure and page references.
4. Normalize extracted content.
5. Chunk into semantically useful units.
6. Generate embeddings.
7. Index chunks for lexical and vector retrieval.
8. Store source ↔ chunk relationships.
9. Record ingestion status and errors.

Re-ingesting an unchanged document must not create duplicates. **Critical:** every retrieved chunk must be traceable to its original document and location.

### B. Hybrid retrieval engine

| Method | Purpose |
|---|---|
| BM25 | Exact terminology, keywords, technical phrases |
| Dense retrieval | Semantic similarity and contextual matching |
| Hybrid | Combines lexical and semantic relevance |

Do not assume semantic similarity means relevance. Pipeline:
1. Analyze the query.
2. Retrieve candidates via BM25.
3. Retrieve candidates via dense vector search.
4. Fuse with a justified method (e.g. Reciprocal Rank Fusion).
5. Deduplicate.
6. Apply metadata filters.
7. Rerank.
8. Return ranked evidence with source references.

### C. Embedding and reranking models (real ML engineering)

- Embedding model is configurable, not hardcoded; support experiments across pretrained models.
- Reranking layer supports cross-encoder models.
- Implement and compare: baseline dense; BM25; hybrid; hybrid + cross-encoder rerank.
- Provide a mechanism to fine-tune/adapt the reranker on domain query–passage relevance pairs.
- **Do not assume fine-tuning helps.** First establish a credible baseline, build a labelled dataset, and determine whether adaptation improves *held-out* performance.
- Metrics: Recall@K, Precision@K, MRR, nDCG@K, retrieval latency, embedding generation cost.
- Evaluation data must be separate from training data.

### D. Research and reasoning engine

LangGraph orchestrates research workflows. The workflow must be explicit and inspectable, not opaque agent calls. Stages:
1. Query understanding — objective, subject, constraints, expected answer.
2. Research planning — what information must be found.
3. Evidence retrieval.
4. Evidence assessment — is the evidence sufficient?
5. Additional research — reformulate and retrieve if needed.
6. Synthesis.
7. Verification — important claims supported by retrieved sources.
8. Response generation — citations and limitations.

Bounded iteration only: no indefinite searching or purposeless tool calls. A small bounded loop of explicit nodes is preferred over a complicated multi-agent architecture.

### E. Temporal and contextual intelligence

Documents can be different versions of the same information (2022 paper, 2024 revision, 2025 report citing both). Preserve publication and effective dates where available. Eventually support: what changed between versions; which source is more recent; was a statement applicable at a date; which documents contradict each other.

Publication, effective, revision and ingestion dates are distinct and not interchangeable. Do not claim temporal correctness when metadata is missing or ambiguous.

### F. Evidence-backed response generation

Support: clear answers; inline citations; expandable supporting evidence; source document references; page/section references; conflicting evidence; explicit uncertainty; distinction between retrieved facts and model-generated interpretation.

A citation must refer to an actual retrieved source location. Never invent citations, and never treat an LLM's assertion that a source supports a claim as proof.

### G. Evaluation and observability

Use RAGAS where appropriate, plus custom evaluation.

| Dimension | Measures |
|---|---|
| Context precision | Retrieved material is relevant |
| Context recall | Relevant evidence was retrieved |
| Faithfulness | Answer supported by context |
| Answer relevance | Answer addresses the question |
| Citation accuracy | Cited sources support the claims |
| Temporal accuracy | Date-sensitive answers use the right version |
| Latency | Workflow duration |

Langfuse for LLM tracing and workflow observability. MLflow for model experiments and comparisons. Distinct responsibilities — don't duplicate experiment tracking.

## 1.4 User journey

1. Create a research workspace.
2. Add knowledge sources (upload or URL); Mezvuat extracts and indexes.
3. Explore the knowledge base (documents, metadata, ingestion status).
4. Ask a research question or objective.
5. Mezvuat plans, performs hybrid retrieval, reranks.
6. Mezvuat reasons; bounded additional retrieval if needed; synthesizes and verifies.
7. Receive a cited answer with passages, uncertainty and document dates.
8. Feedback and evaluation results are recorded for analysis and improvement.

## 1.5 Internal architecture

Modular monolith with clean module boundaries. No microservices until a workload justifies them.

| Layer | Responsibility |
|---|---|
| Frontend | Workspaces, document management, chat, evidence viewer |
| API | Auth, workspace management, request validation |
| Ingestion | Parsing, normalization, chunking, indexing |
| Retrieval | BM25, vector search, fusion, reranking |
| Orchestration | LangGraph research workflow |
| ML | Embedding/reranking interfaces, training, evaluation |
| Persistence | PostgreSQL, Qdrant, document storage |
| Inference | Provider-independent LLM interface |
| Evaluation | RAG metrics, benchmarks, feedback |
| Observability | Tracing, logs, operational metrics |

Proposed stack: Python; FastAPI; Next.js + React + TypeScript; LangGraph; PyTorch; Hugging Face; sentence-transformers; Qdrant; PostgreSQL; PostgreSQL FTS initially or a dedicated BM25 implementation; RAGAS + custom eval; Langfuse; MLflow; Docker.

LangChain components may be used where they add value, but LangGraph owns the explicit workflow. No overlapping orchestration abstractions.

## 1.6 Out of scope (initial)

General internet search engine; unrestricted autonomous agent; generic ChatGPT clone; thin LLM wrapper over uploads; claims of factual accuracy without evidence; foundation-model fine-tuning framework; multi-agent swarm; SaaS billing/subscriptions.

---

# 2. Product Two: ModelMesh

*One interface. Multiple models. Intelligent inference.*

## 2.1 Product vision

ModelMesh is a self-hostable, provider-independent LLM inference gateway between AI applications and model providers. Applications don't need to know how providers authenticate, format requests/responses, enforce limits or report usage — they use one unified API.

Not just another API proxy: **the defining feature is intelligent model selection based on measured quality, cost, latency and reliability** — deciding which available model suits a request under a set of constraints. This is ModelMesh's primary ML engineering component.

## 2.2 Who is it for?

Developers building LLM apps; small teams using multiple providers; developers needing cost controls; self-hosters mixing local and cloud models; teams wanting inference visibility; apps needing provider independence and resilience.

Mezvuat is the first customer, but ModelMesh must never be architecturally dependent on Mezvuat.

## 2.3 Core capabilities

### A. Unified inference API

OpenAI-compatible wherever practical. A request can specify: model or routing policy; messages/prompt; temperature and generation params; max output tokens; optional cost and latency constraints; streaming; request metadata. ModelMesh translates to provider format and normalizes responses into a common schema. Applications need no provider-specific code.

### B. Provider abstraction layer

Independent adapters behind a common interface. Initial targets (not all required immediately): Gemini, Groq, OpenRouter, local models via a compatible runtime such as Ollama.

Adapter responsibilities: authentication; request translation; response normalization; token usage extraction; streaming where available; error classification; timeout handling; provider-specific rate-limit handling.

Credentials securely managed, never exposed to clients. Provider-specific behaviour stays inside adapters.

### C. Intelligent model router (core ML differentiator)

Static rules (use A; fall back to B; cheap model for short prompts) are useful but not intelligent. ModelMesh implements a **learned** router considering:

- **Request characteristics:** prompt length; estimated input tokens; task category; complexity indicators; structured-output requirements; context requirements; streaming requirements.
- **Model characteristics:** historical task-specific quality; input/output token pricing; latency distribution; reliability/error rate; context window; capability compatibility.
- **User constraints:** max estimated cost; latency budget; minimum quality; allowed providers/models; required capabilities.

#### Training pipeline
1. Build a representative request dataset across task categories.
2. Benchmark each compatible model on the same workload.
3. Evaluate quality with task-appropriate metrics and, where appropriate, human-reviewed labels.
4. Record costs, latency and failure rates.
5. Construct training examples from these observations.
6. Train a routing classifier or ranking model.
7. Compare against a static rules-based baseline.
8. Evaluate on held-out queries and workload distributions.
9. Integrate into the routing engine.
10. Monitor and retrain when sufficient new evidence exists.

No RL or complicated neural architecture needed initially; a gradient-boosted classifier/ranker may beat a neural net. Choose based on experimental results, not resume keywords.

#### Routing objective
Policies are explicit, not hidden in an arbitrary weighted score. Examples: minimize cost subject to a quality threshold; minimize latency within a cost budget; prefer reliability over cost; best model regardless of cost.

Formalization:

  min over m ∈ M of Cost(m, q)
  subject to  Q(m, q) ≥ Q_min  and  L(m, q) ≤ L_max

Not every request has enough information to guarantee quality or latency — communicate estimates and uncertainty honestly.

### D. Fallback and reliability

Health checks; timeouts; retry policies; exponential backoff with jitter; circuit breakers; alternative provider selection; error normalization; fallback execution. Retries distinguish transient from permanent errors (never retry an invalid API key). Fallback must respect application constraints — never silently violate quality, privacy or budget policy.

### E. Semantic caching (Redis)

Exact request caching; semantic-similarity caching where appropriate; expiration; invalidation; model- and configuration-aware keys; tenant and user isolation.

Similar questions don't necessarily have identical answers. Cache reuse must consider model, system prompt, generation settings, context and relevant metadata. For Mezvuat, retrieval context and source versions are part of cache correctness: a response grounded in an old document version must never be returned as if it reflects newer evidence. Caching configurable per application and request type.

### F. Cost and usage management

Track: provider; model; input/output tokens; estimated cost; provider-reported usage where available; latency; success/failure; cache hits; routing decision; fallback events. Configurable API-key budgets, usage limits, rate limits, request quotas, provider restrictions. Estimated cost is not billing data — keep calculated vs provider-reported figures distinct.

### G. Observability

OpenTelemetry for traces and metrics. Every request has a traceable lifecycle: incoming request → policy evaluation → router decision → cache lookup → provider request → response normalization → usage accounting → final response. Don't indiscriminately log prompts or keys. Langfuse may be used for app-level LLM tracing where useful; OpenTelemetry remains the infrastructure standard.

### H. Developer experience

API-key creation/revocation; provider configuration; model availability; routing policy configuration; usage dashboard; cost analytics; latency/reliability stats; request traces; benchmark comparison. A minimal dashboard suffices; the API and its docs are the product foundation.

## 2.4 Request lifecycle

1. App sends inference request.
2. Authentication and validation (API key, format, permissions, usage limits).
3. Policy evaluation (allowed providers/models, budget, performance requirements).
4. Cache evaluation (safe, valid cached response?).
5. Intelligent routing.
6. Inference execution via adapter.
7. Failure handling (retry/fallback policies).
8. Response normalization (with usage and routing metadata).
9. Accounting and observability.
10. Feedback and model improvement (labelled outcomes stored for router training).

## 2.5 Internal architecture

| Layer | Responsibility |
|---|---|
| API gateway | Unified inference endpoints |
| Authentication | API keys, access, quotas |
| Policy engine | Budgets, restrictions, constraints |
| Router | Model selection and routing policies |
| ML engine | Training, inference, router evaluation |
| Provider adapters | Provider-specific execution |
| Cache | Exact and semantic caching |
| Reliability | Retries, fallback, circuit breakers |
| Accounting | Token usage and cost records |
| Observability | Traces, metrics, structured logs |
| Persistence | PostgreSQL and Redis |

Proposed stack: Python; FastAPI; scikit-learn (optionally PyTorch); provider SDKs / compatible HTTP APIs; Redis; PostgreSQL; MLflow; OpenTelemetry; Langfuse where useful; Pytest, HTTPX, Locust; Docker.

## 2.6 Out of scope (initial)

A new foundation model; a model training platform; a replacement for every gateway; a Kubernetes control plane; a cloud hosting provider; billing SaaS with payments; an unrestricted autonomous model selector; a router claiming quality guarantees without supporting evaluation data.

---

# 3. How Mezvuat AI and ModelMesh interact

## 3.1 Fundamental relationship

- **Mezvuat** owns intelligence and the research workflow: understands the question, retrieves evidence, plans research, orchestrates tasks, generates the final answer.
- **ModelMesh** owns inference execution and optimization: chooses models, executes requests, handles providers, optimizes performance, records inference telemetry.
- They communicate through a **versioned inference API contract**.

Mezvuat must not know how ModelMesh selects providers. ModelMesh must not know how Mezvuat retrieves documents or orchestrates research.

## 3.2 End-to-end example

User: *"Compare the latest findings in these three research papers and explain where their conclusions disagree."*

1. User → Mezvuat API.
2. LangGraph planner identifies a three-document comparison and extracts objectives.
3. Hybrid retrieval finds relevant passages from all three papers.
4. Mezvuat → ModelMesh inference request to analyze the evidence.
5. ModelMesh router evaluates task, providers, quality, latency, cost constraints.
6. Selected model receives the normalized request via its adapter.
7. Response returns with normalized usage and execution metadata.
8. Mezvuat verifies the response against the evidence; decides if more research is needed.
9. Final comparison with citations, disagreements and limitations is shown to the user.
10. Mezvuat records research quality; ModelMesh records inference performance and routing outcomes.

## 3.3 Request types

| Mezvuat operation | ModelMesh execution requirement |
|---|---|
| Query classification | Low-latency, inexpensive |
| Query rewriting | Efficient instruction-following |
| Research planning | More capable reasoning |
| Evidence synthesis | Strong instruction-following and context handling |
| Citation verification | Reliable structured output |
| Final response | Quality-oriented generation |

These are routing preferences, not hardcoded provider assignments. Selection is based on benchmarked capabilities and application policies. Mezvuat can require particular capabilities or structured-output guarantees.

## 3.4 Improvement loops

**Loop A — Mezvuat:** run research tasks → evaluate retrieval and answer quality → categorize failures → add examples to the retrieval eval dataset → run embedding/reranking experiments → deploy better configurations → measure on the same held-out benchmark.

**Loop B — ModelMesh:** receive requests → record task characteristics and outcomes → benchmark/eval results provide quality labels → record cost, latency, reliability → build router training data → retrain and evaluate vs baseline → deploy improved routing.

ModelMesh must not assume HTTP 200 means a good answer. Mezvuat owns application-level quality evaluation; ModelMesh owns execution-level measurements. Quality labels are shared through an explicit feedback interface with appropriate identifiers and privacy controls. Neither system may silently modify its production model based on unverified user feedback.

---

# 4. Integration contract and ownership boundaries

## 4.1 Communication contract

ModelMesh exposes a versioned API. Suggested endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Unified inference |
| `POST /v1/embeddings` | Embedding inference, if supported |
| `GET /v1/models` | Available models |
| `GET /v1/health` | Service health |
| `GET /v1/providers` | Provider status |
| `GET /v1/usage` | Usage information |
| `GET /v1/requests/{id}` | Request execution metadata |

Mezvuat uses a dedicated inference client — never ModelMesh's database or internal modules.

Illustrative (not final) request:

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "Analyze the supplied research evidence."},
    {"role": "user", "content": "Compare the findings."}
  ],
  "routing": {"policy": "quality_balanced", "max_cost": 0.02, "max_latency_ms": 10000},
  "metadata": {"application": "mezvuat", "task_type": "evidence_synthesis"}
}
```

Agents define and validate the actual schemas, including units, optional fields, error formats and streaming behaviour. The working draft lives in `docs/contract/`.

## 4.2 Responsibility boundaries

| Responsibility | Mezvuat | ModelMesh |
|---|---|---|
| Document ingestion | ✅ | |
| Chunking | ✅ | |
| Embedding evaluation | ✅ | |
| Hybrid retrieval | ✅ | |
| Research orchestration | ✅ | |
| Citation verification | ✅ | |
| Provider adapters | | ✅ |
| Model routing | | ✅ |
| Provider fallback | | ✅ |
| Inference caching | | ✅ |
| Usage accounting | | ✅ |
| Application answer quality | ✅ | |
| Infrastructure observability | | ✅ |
| Router training | | ✅ |
| Retrieval model training | ✅ | |

ModelMesh may offer embedding inference, but Mezvuat owns embedding-model selection, retrieval evaluation and indexing strategy.

## 4.3 Failure handling

- ModelMesh unavailable → Mezvuat uses a configured direct-provider fallback (proves loose coupling).
- One ModelMesh provider fails → ModelMesh tries an eligible alternative.
- Qdrant unavailable → Mezvuat reports retrieval unavailability; never fabricates an answer.
- Evaluation service fails → the user-facing workflow need not fail.
- Router can't decide reliably → ModelMesh uses a documented safe fallback policy.

---

# 5. Shared engineering principles

## 5.1 Independent repositories

```
projects/
├── mezvuat-ai/   (frontend, backend, retrieval, orchestration, ml, evaluation, documentation)
└── modelmesh/    (api, providers, routing, ml, caching, observability, documentation)
```

No shared repository. Independent dependencies, environment config, tests, Dockerfiles and release processes.

## 5.2 Engineering quality

Both products: type-checked, validated API contracts; structured logging; centralized config; env-based secrets; unit and integration tests; Docker-based local dev; CI; DB migrations; error handling; reproducible ML experiments; docs explaining architectural decisions.

PostgreSQL for persistent data; Redis for ephemeral caching and rate-limit state; Qdrant for vectors in Mezvuat. Avoid unnecessary technology duplication.

## 5.3 Security and privacy

Never commit or log secrets. Scope API keys and workspace data. Prevent cross-workspace retrieval and cache leakage. Sanitize uploads and extracted content. Treat document contents as untrusted data, not instructions; protect against prompt injection in retrieved documents. Apply provider restrictions to sensitive requests. Define data retention and deletion. Never send documents or prompts to external providers without the configured application's authorization.

---

# 6. Definition of done

## Mezvuat AI
- [ ] Users can create and manage research workspaces.
- [ ] Documents can be ingested, parsed, indexed and traced to original sources.
- [ ] Hybrid retrieval works (lexical + dense).
- [ ] Reranking can be enabled and evaluated independently.
- [ ] A labelled retrieval evaluation dataset exists.
- [ ] Retrieval experiments are reproducible.
- [ ] LangGraph workflow supports bounded research iterations.
- [ ] Answers include verifiable source citations.
- [ ] Document version and temporal metadata are preserved.
- [ ] Application-level quality evaluation is available.
- [ ] Can use ModelMesh and a direct-provider fallback.
- [ ] Runs independently using Docker.

## ModelMesh
- [ ] Unified, documented inference API works.
- [ ] Multiple provider adapters implemented and tested.
- [ ] Provider responses normalized.
- [ ] Routing engine supports explicit policies and constraints.
- [ ] Trained routing model compared against a rules-based baseline.
- [ ] Routing experiments use held-out evaluation data.
- [ ] Fallback, retry and circuit-breaker behaviour tested.
- [ ] Caching is safe and tenant-aware.
- [ ] Token usage, cost, latency and reliability recorded.
- [ ] API-key management and rate limits work.
- [ ] OpenTelemetry traces available.
- [ ] Runs independently using Docker.

## Combined integration
- [ ] Mezvuat uses ModelMesh through its public API.
- [ ] Both apps remain independently deployable.
- [ ] Shared request IDs enable end-to-end tracing.
- [ ] Application quality results can be associated with inference executions.
- [ ] A complete research workflow works through ModelMesh.
- [ ] Direct-provider fallback works when ModelMesh is unavailable.
- [ ] A combined benchmark report demonstrates quality, cost and latency trade-offs.

These are completion criteria, not a demand that every advanced feature exist before either product can run.

---

# 7. Instructions to the coding agents

## Shared implementation mandate

You are implementing one of two independent but integrated AI products: Mezvuat AI and ModelMesh. Read the complete product specification before implementation. First inspect the existing repository, tooling, dependencies, conventions and environment. Do not assume a clean repository or overwrite existing work. Preserve architectural independence. Mezvuat owns research orchestration, document ingestion, retrieval, reranking, citations and application-level evaluation. ModelMesh owns the unified inference API, provider adapters, routing, fallback, caching, usage accounting and infrastructure observability. Integrate only through documented, versioned public APIs. Do not access the other product's database or internal modules. Build real, testable functionality; do not substitute mock implementations for production paths without clearly isolating them. Establish baselines before claiming ML improvements. Keep training, validation and test datasets separate. Do not fabricate benchmark results. Keep scope disciplined; do not add Kubernetes, multi-cloud, billing, complex multi-agent systems or unnecessary services without explicit justification. Add tests, migrations, Docker support, CI and documentation. Record assumptions, architectural decisions, limitations and unfinished work. Never claim completion without executing and reporting relevant verification. If a dependency, API key, external service or dataset is unavailable, isolate the blocker and provide a reproducible alternative. The products must work independently as well as together.

## Per-agent ownership

- **Mezvuat agent:** owns Mezvuat's repository and implementation. Implements the ModelMesh client against the agreed public contract; must not modify ModelMesh's internals.
- **ModelMesh agent:** owns ModelMesh's repository and implementation. Publishes the API contract and integration docs; must not modify Mezvuat's internals.
- **Integration:** agree on request/response schemas before implementing integration clients. On a contract mismatch, report it and propose a coordinated change — never change both systems independently.

---

# 8. Open decisions requiring ADRs

1. Exact embedding and reranking models.
2. Qdrant vs an alternative — only with a compelling technical reason.
3. PostgreSQL full-text search vs a dedicated BM25 engine.
4. Initial learned routing algorithm.
5. Collecting trustworthy quality labels for ModelMesh.
6. Semantic cache correctness and isolation strategy.
7. Streaming and structured-output compatibility across providers.
8. Correlating application evaluation records with gateway traces.
9. Document parsing strategy and storage backend.
10. Deployment architecture and cloud provider.

Document decisions, explain trade-offs, and don't add dependencies just because they're popular.

---

# 9. Product identity

**Mezvuat AI** — AI application engineering + applied ML. A research intelligence platform that retrieves, reasons and produces verifiable knowledge from complex sources. Demonstrates RAG/retrieval engineering, embedding and reranking experiments, agentic orchestration, PyTorch and Hugging Face, evaluation-driven development, evidence-grounded generation.

**ModelMesh** — ML systems + AI infrastructure. An intelligent inference gateway that selects, executes and optimizes model requests across providers. Demonstrates ML-based routing, model serving and provider abstraction, cost and latency optimization, caching and reliability, observability, production-oriented backend engineering.

**Together:** an AI application and the intelligent inference infrastructure powering it. One shows how to build intelligence; the other shows how to operate it efficiently.

**Process rule from the product owner:** don't start by generating code from the descriptions. Specs → repository inspection → architecture decisions → API contracts → *then* authorized implementation. Especially for learned routing and retrieval: real ML work, not API integrations dressed up as ML.
