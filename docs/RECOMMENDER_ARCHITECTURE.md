# Recommender Architecture

## Change Scope

| File | Change | Dependency impact |
| --- | --- | --- |
| `api.py` | Add user-aware ranking, interaction APIs, benchmark and version endpoints | Existing query parameters and response fields remain valid |
| `frontend/app.js` | Add anonymous user identity and interaction telemetry | No new frontend dependency |
| `recommender/interaction_store.py` | Replace JSONL persistence with transactional SQLite storage | Python standard library only |
| `recommender/user_profile.py` | Derive recency, affinity, completion, and progression features | Reads the interaction store |
| `recommender/collaborative_filtering.py` | Train practical implicit matrix factorization scores | Uses existing NumPy, pandas, and scikit-learn |
| `recommender/ranking.py` | Add configurable behavioral and collaborative weights | No new dependency |
| `recommender/evaluation.py` | Add coverage, diversity, and popularity-bias metrics | Uses existing NumPy and pandas |
| `recommender/semantic_search.py` | Track artifact metadata and avoid unnecessary index rewrites | No new dependency |
| `recommender/explainability.py` | Explain behavioral and collaborative signals | No new dependency |
| `recommender/experiments.py` | Compare five recommendation strategies | Uses existing recommender components |
| `recommender/model_registry.py` | Fingerprint data, models, indexes, and ranker configuration | Python standard library only |
| `scripts/run_benchmark.py` | Generate reproducible JSON and HTML reports | No new dependency |

## Before

```text
Browser
  -> GET /recommend
      -> TF-IDF or SentenceTransformer/FAISS
      -> static request preferences
      -> fixed hybrid ranker
  -> POST /users/{id}/events
      -> process-local profile
      -> JSONL append log

Collaborative filtering module (not connected)
Evaluation functions (manual relevance lists only)
```

## After

```text
Browser
  -> stable anonymous user_id
  -> POST /users/{id}/events
      -> validated event schema
      -> SQLite WAL interaction log
      -> user history and training export

GET /recommend?user_id=...
  -> semantic retrieval ---------------------------+
  -> explicit request preferences -----------------|
  -> recent and historical behavior ---------------+-> configurable ranker
  -> category affinity and completion progression --|
  -> implicit matrix factorization ----------------+
      -> ranked response + explanations + model versions

Offline experiment runner
  -> TF-IDF
  -> semantic search
  -> hybrid
  -> collaborative filtering
  -> hybrid + collaborative filtering
      -> Precision@K, Recall@K, MAP, NDCG
      -> coverage, diversity, popularity bias
      -> JSON report and HTML dashboard
```

## Collaborative Filtering Choice

The implementation uses weighted implicit matrix factorization through truncated
SVD. LightFM is a strong option when side features and sufficient interaction
volume are available, but it adds a compiled dependency. Neural collaborative
filtering requires substantially more data, training infrastructure, and
monitoring than this repository currently has. Truncated SVD is deterministic,
deployable with the existing stack, and provides a clean upgrade path to LightFM
or an implicit-ALS service without changing the ranking API.

## Data Contract

Interaction events contain:

```text
event_id, user_id, course_id, course, event_type, weight,
category, sub_category, platform, level, session_id, source,
metadata_json, occurred_at, created_at
```

Supported event types are `view`, `click`, `save`, `enroll`, and `complete`.
The event table is append-only and indexed for user-history and model-training
queries.

## Reproducibility

Every recommendation response exposes model metadata for:

- sentence-transformer name
- dataset fingerprint
- embedding fingerprint
- FAISS index fingerprint
- ranking configuration fingerprint
- recommendation version

Benchmark reports include the same metadata, configuration, random seed, and
generation timestamp.
