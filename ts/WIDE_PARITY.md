# Wide Parity: Python vs TypeScript Packages

Cross-package parity analysis across all TypeScript packages vs the Python source.
Last updated: 2026-03-06

---

## Quick Summary

| Package | Coverage | Status |
|---------|----------|--------|
| **kernel** | ~70% | Documented in [PARITY.md](./PARITY.md) |
| **agents** (core) | ~85% | Missing infra-dependent pieces |
| **agents** (tools) | ~70% | 14/24 tools ported |
| **models** (schemas) | ~94% | All schemas present |
| **models** (methods) | ~40% | Query/mutation methods mostly missing |
| **runtime/providers** | ~35% | Core LLM providers present, multimodal missing |
| **websocket/api** | ~65% | Core endpoints present, 10+ modules missing |
| **security** | ~75% | Crypto/keys ported; auth providers missing |
| **storage** | **0%** | Entire layer absent |
| **config** | **0%** | No env loading or settings persistence |
| **metadata** | ~20% | Only message types ported |
| **chat** | ~5% | Core loop only |

**Estimated overall TypeScript parity: ~50%**

---

## 1. Kernel (`packages/kernel`)

See **[PARITY.md](./PARITY.md)** for the full kernel gap list. Summary of open items:

| Gap | Status |
|-----|--------|
| #3 Graph validation pipeline (filter invalid edges, type compat) | Open |
| #5 zip_all stickiness uses streaming analysis | Open |
| #9 OutputUpdate messages per value | Open |
| #10 Multi-edge list type validation | Open |
| #11 Controlled node lifecycle (save/restore, response_future, metadata) | Open |
| #13 Node finalization in finally block | **Fixed** |
| #14 Graph.fromDict() error recovery | Open |
| #15 Edge counter updates at all lifecycle points | Open |

---

## 2. Agents (`packages/agents`)

### Core Agent System — 85% complete

| Python File | TS File | Status |
|-------------|---------|--------|
| `agent.py` | `agent.ts` | ✅ Full — skills, planning/reasoning models, pre-defined task |
| `base_agent.py` | `base-agent.ts` | ✅ Full |
| `simple_agent.py` | `simple-agent.ts` | ✅ Full |
| `task_planner.py` | `task-planner.ts` | ✅ Full — includes `reasoningModel` |
| `step_executor.py` | `step-executor.ts` | ✅ Full |
| `task_executor.py` | `task-executor.ts` | ✅ Full |
| `agent_executor.py` | `agent-executor.ts` | ✅ Full |
| `agent_evaluator.py` | — | ❌ Python-only CLI/batch runner |
| `docker_runner.py` | — | ❌ Infrastructure-specific |
| `graph_planner.py` | — | ❌ Requires networkx + workflow infra |
| `vibecoding.py` | — | ❌ Workflow-specific |
| `wrap_generators_parallel.py` | — | ❌ Simple utility, can be inlined |

### Agent Tools — 70% complete (14/24 ported)

| Python Tool | TS Tool | Status |
|-------------|---------|--------|
| `base.py` | `base-tool.ts` | ✅ |
| `browser_tools.py` | `browser-tools.ts` | ✅ |
| `chroma_tools.py` | `chroma-tools.ts` | ✅ |
| `code_tools.py` | `code-tools.ts` | ✅ |
| `email_tools.py` | `email-tools.ts` | ✅ |
| `filesystem_tools.py` | `filesystem-tools.ts` | ✅ |
| `finish_step_tool.py` | `finish-step-tool.ts` | ✅ |
| `google_tools.py` | `google-tools.ts` | ✅ |
| `http_tools.py` | `http-tools.ts` | ✅ |
| `math_tools.py` | `math-tools.ts` | ✅ |
| `mcp_tools.py` | `mcp-tools.ts` | ✅ |
| `openai_tools.py` | `openai-tools.ts` | ✅ |
| `pdf_tools.py` | `pdf-tools.ts` | ✅ |
| `serp_tools.py` | `search-tools.ts` | ⚠️ Partial — SERP provider abstraction missing |
| `asset_tools.py` | — | ❌ Needs `context.create_asset()` |
| `control_tool.py` | — | ❌ Workflow infra dependency |
| `help_tools.py` | — | ❌ Needs semantic search infra |
| `model_tools.py` | — | ❌ Needs model registry API |
| `node_tool.py` | — | ❌ Needs BaseNode system |
| `workspace_tools.py` | — | ❌ Needs agent workspace |
| `_remove_base64_images.py` | — | ❌ Auxiliary helper |

### Top-level `nodetool/tools/` — 0% ported
All 9 files (agent_tools, asset_tools, collection_tools, hf_tools, job_tools, model_tools, node_tools, storage_tools, workflow_tools) depend on the Node framework layer and are intentionally out of scope.

---

## 3. Models (`packages/models`)

### Schema Parity — ~94% complete

All 12+ domain models have schemas ported. The base infrastructure (DBModel, condition-builder, SQLite adapter, memory adapter) is fully ported.

### Query/Mutation Method Parity — ~40% complete

Every domain model is **missing most of its static query and mutation methods**:

| Model | Missing Methods |
|-------|----------------|
| `Asset` | `paginate()`, `find()`, `get_children()`, `search_assets_global()`, `get_asset_path_info()`, `get_assets_recursive()` |
| `Job` | `paginate()`, `find()`, `claim()`, `release()`, `update_heartbeat()`, `acquire_with_cas()`, `mark_*()`, `is_*()` |
| `Workflow` | `paginate()`, `find()`, `paginate_tools()`, `find_by_tool_name()`, `has_trigger_nodes()`, `get_api_graph()`, `from_dict()` |
| `Message` | `paginate()`, `create()` with content conversion, deserializer validators |
| `Thread` | `find()`, `create()`, `paginate()` |
| `Secret` | `find()`, `list_for_user()`, `list_all()`, `get_decrypted_value()`, `upsert_encrypted()`, full encryption integration |
| `OAuthCredential` | `create_encrypted()`, `find_by_account()`, `list_for_user_and_provider()`, `get_decrypted_*()`, `update_tokens()`, encryption integration |
| `Prediction` | `create()`, `find()`, `paginate()`, `aggregate_by_*()` |
| `RunEvent` | `create()`, `get_next_seq()`, `append_event()`, `get_events()`, `get_last_event()` |
| `RunNodeState` | `get_node_state()`, `get_or_create()`, status management |
| `RunLease` | `acquire()`, `renew()`, `release()`, `is_expired()` |

### Not Ported (Python-specific or complex infra)

| Python File | Reason |
|-------------|--------|
| `postgres_adapter.py` | PostgreSQL-specific |
| `supabase_adapter.py` | Supabase-specific |
| `migrations.py` | Migration system — no TS equivalent needed |
| `run_inbox_message.py` | Durable inbox; not yet needed |
| `trigger_input.py` | Trigger system; not yet needed |
| `workflow_version.py` | Workflow versioning; not yet needed |

---

## 4. Runtime/Providers (`packages/runtime`)

### Provider Coverage — ~35% average LOC coverage

| Provider | Python LOC | TS LOC | Notes |
|----------|-----------|--------|-------|
| OpenAI | 2,315 | 1,105 | Missing embeddings, speech, image gen, vision |
| Anthropic | 760 | 700 | Missing vision support |
| Gemini | 1,683 | 537 | Missing multimodal (600+ LOC) |
| Ollama | 929 | 433 | Simplified but functional |
| Llama | 1,146 | 489 | Basic only |
| Groq | 166 | 85 | Minimal |
| Mistral | 534 | 96 | Minimal |
| OpenRouter | 753 | 95 | Minimal |
| Together | 472 | 89 | Minimal |
| Cerebras | 464 | 85 | Minimal |
| LM Studio | 408 | 95 | Basic |
| vLLM | 416 | 99 | Basic |

### Providers Not Ported

| Provider | Size | Type |
|----------|------|------|
| HuggingFace | 1,443 LOC | LLM — complex inference API |
| Llama Server Manager | 824 LOC | Infrastructure |
| KIE, MiniMax, ZAI, AIME | ~2,500 LOC | Specialized/regional |
| ComfyUI (3 variants) | ~1,390 LOC | Local image generation |
| Meshy, Rodin | ~884 LOC | 3D generation |

### Critical Missing Capabilities

- **Embeddings** — zero implementation in any TS provider
- **Vision/Multimodal** — Claude Vision, GPT-4V, Gemini multimodal all missing
- **Text-to-Image/Video** — all ComfyUI, Meshy, Rodin missing
- **Chat infrastructure** — CLI, persistence, commands: ~5% ported (core loop only)

---

## 5. WebSocket/API (`packages/websocket`)

### Endpoint Coverage — ~65%

| Python Module | TS Handler | Status |
|---------------|-----------|--------|
| `workflow.py` | `http-api.ts` | ⚠️ Partial — missing autosave, generate_name |
| `job.py` | `http-api.ts` | ⚠️ Partial — missing running/all |
| `asset.py` | `http-api.ts` | ⚠️ Partial — missing search, packages |
| `message.py` | `http-api.ts` | ⚠️ Partial — missing delete |
| `thread.py` | `http-api.ts` | ⚠️ Partial — missing update |
| `settings.py` | `http-api.ts` | ⚠️ Partial — missing GET/PUT /settings |
| `node.py` | `http-api.ts` | ⚠️ Partial — missing replicate_status |
| `users.py` | `http-api.ts` | ⚠️ Partial — missing username validation |
| `model.py` | `models-api.ts` | ⚠️ Partial — missing HuggingFace cache check |
| `workspace.py` | `workspace-api.ts` | ✅ Full |
| `cost.py` | `cost-api.ts` | ✅ Full |
| `skills.py` | `skills-api.ts` | ✅ Full |
| `font.py` | `skills-api.ts` | ✅ Full |
| `openai.py` | `openai-api.ts` | ✅ Full |
| `oauth.py` | `oauth-api.ts` | ✅ Full |
| `admin_secrets.py` | — | ❌ Bulk secret import |
| `collection.py` | — | ❌ Asset collections |
| `debug.py` | — | ❌ Debug bundle export |
| `file.py` | — | ❌ File browser / download |
| `memory.py` | — | ❌ Model load/unload lifecycle |
| `storage.py` | — | ❌ Key-value storage API |
| `vibecoding.py` | — | ❌ HTML app generation |
| `mcp_server.py` | — | ❌ MCP protocol server |
| `fal_schema.py` | — | ❌ FAL dynamic schema resolution |
| `kie_schema.py` | — | ❌ KIE dynamic schema resolution |
| `replicate_schema.py` | — | ❌ Replicate dynamic schema resolution |

---

## 6. Security (`packages/security`, `packages/auth`)

| Python File | TS File | Status |
|-------------|---------|--------|
| `crypto.py` | `security/src/crypto.ts` | ✅ Ported |
| `master_key.py` | `security/src/master-key.ts` | ✅ Ported |
| `secret_helper.py` | `security/src/secret-helper.ts` | ✅ Ported |
| `auth_provider.py` | `auth/src/auth-provider.ts` | ✅ Ported |
| `http_auth.py` | — | ❌ HTTP auth middleware |
| `admin_auth.py` | — | ❌ Admin authentication |
| `user_manager.py` | — | ❌ User management / RBAC |
| `startup_checks.py` | — | ❌ Startup validation |
| `providers/local.py` | — | ❌ Local auth provider |
| `providers/multi_user.py` | — | ❌ Multi-user auth |
| `providers/static_token.py` | `auth/src/providers/static-token-provider.ts` | ✅ |
| `providers/supabase.py` | — | ❌ Supabase auth |
| `aws_secrets_util.py` | — | ❌ AWS Secrets CLI |

---

## 7. Entirely Absent in TypeScript

### Storage layer — 0%
| Python Module | What It Does |
|---------------|-------------|
| `storage/abstract_storage.py` | Core storage abstraction (get_url, upload, download, delete) |
| `storage/file_storage.py` | Local filesystem backend |
| `storage/s3_storage.py` | AWS S3 backend with multipart upload |
| `storage/supabase_storage.py` | Supabase cloud storage |
| `storage/memory_storage.py` | In-memory backend (useful for tests) |
| `storage/abstract_node_cache.py` | Node result caching abstraction |
| `storage/memory_node_cache.py` | In-memory node cache |
| `storage/memcache_node_cache.py` | Distributed Memcache node cache |
| `storage/memory_uri_cache.py` | URI generation and asset caching |

### Config layer — 0%
| Python Module | What It Does |
|---------------|-------------|
| `config/environment.py` | Env loading, DEFAULT_ENV dict, hierarchy |
| `config/settings.py` | Settings persistence, `register_setting()` |
| `config/logging_config.py` | Logging setup, Sentry integration |
| `config/env_diagnostics.py` | Environment validation |

### Metadata reflection — 80% missing
| Python Module | What It Does |
|---------------|-------------|
| `metadata/type_metadata.py` | Type reflection utilities |
| `metadata/node_metadata.py` | Node class introspection |
| `metadata/typecheck.py` | Type validation |

### Messaging layer — 63% missing
| Python Module | What It Does |
|---------------|-------------|
| `messaging/help_message_processor.py` | Node/example semantic search |
| `messaging/chat_workflow_message_processor.py` | Chat-based workflow orchestration |
| `messaging/workflow_message_processor.py` | Workflow execution via chat |
| `messaging/context_packer.py` | Context serialization for prompts |

---

## Recommended Port Priority

### Phase 1 — Unblocks most use cases
1. **Model query methods** — Add `paginate()`, `find()`, `create()` to Asset, Job, Workflow, Message, Thread (the 5 most-used models)
2. **Vision/multimodal** — Add image input support to OpenAI and Anthropic providers
3. **Embeddings** — Add embedding support to at least OpenAI provider

### Phase 2 — Production readiness
4. **Storage layer** — Port `AbstractStorage` + `FileStorage` (minimum for local deployments)
5. **Auth middleware** — Port `http_auth.py` for proper request authentication
6. **Config layer** — Centralized env loading so packages don't reinvent it

### Phase 3 — Feature completeness
7. **File API** — `/api/files/*` endpoints for file browsing/download
8. **Secret/OAuth encryption methods** — `upsert_encrypted()`, `get_decrypted_*()` on Secret and OAuthCredential
9. **Remaining agent tools** — `workspace_tools`, `help_tools`
10. **Additional providers** — HuggingFace, extended provider feature coverage
