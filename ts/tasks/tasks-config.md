# Config Tasks — `packages/config` (new package)

The Python `src/nodetool/config/` module provides centralized environment loading, settings persistence, and logging configuration. TypeScript packages currently read `process.env` directly with no centralized hierarchy.

**This is a greenfield package.** Create `ts/packages/config/` with `src/` and `tests/`.

---

### T-CFG-1 · Environment loader
**Status:** 🔴 open
**Python source:** `config/environment.py`

Python loads config in this hierarchy:
1. `DEFAULT_ENV` dict (hardcoded defaults)
2. Base `.env` file
3. Environment-specific `.env.{NODE_ENV}` file
4. Local override `.env.{NODE_ENV}.local` (gitignored)
5. System environment variables (highest priority)

TypeScript has no equivalent. Each package reads `process.env` directly.

- [ ] **TEST** — Write test: `loadEnvironment()` reads `.env.test` when `NODE_ENV=test`; values from `.env.test.local` override `.env.test`; system env overrides both.
- [ ] **TEST** — Write test: defaults from `DEFAULT_ENV` are applied for missing keys.
- [ ] **TEST** — Write test: `getEnv("MISSING_KEY", "default")` returns `"default"`.
- [ ] **IMPL** — Create `ts/packages/config/src/environment.ts`. Use `dotenv` npm package to load files in hierarchy order. Export `getEnv(key, default?)` and `requireEnv(key)` helpers.

---

### T-CFG-2 · Settings registry
**Status:** 🔴 open
**Python source:** `config/settings.py` — `register_setting()` API for packages to declare their settings; central discovery for UI configuration panels.

- [ ] **TEST** — Write test: `registerSetting({ package, envVar, group, description, isSecret })` adds entry to global registry. `getSettings()` returns all registered settings.
- [ ] **TEST** — Write test: `getSettings()` marks a setting as "configured" when its env var has a non-empty value.
- [ ] **IMPL** — Create `ts/packages/config/src/settings.ts`. Export `registerSetting()`, `getSettings()`, `SettingDescriptor` type.

---

### T-CFG-3 · Logging configuration
**Status:** 🔴 open
**Python source:** `config/logging_config.py` — configures structlog with formatters, log level from env, optional Sentry integration.

- [ ] **TEST** — Write test: `configureLogging({ level: "debug" })` sets minimum log level. DEBUG messages appear; below-level messages suppressed.
- [ ] **TEST** — Write test: `configureLogging({ format: "json" })` emits structured JSON log lines.
- [ ] **IMPL** — Create `ts/packages/config/src/logging.ts`. Use `pino` or a simple wrapper around `console`. Read `LOG_LEVEL` / `NODETOOL_LOG_LEVEL` from env. Optional Sentry integration when `SENTRY_DSN` is set.

---

### T-CFG-4 · Environment diagnostics
**Status:** 🔴 open
**Python source:** `config/env_diagnostics.py` — validates required env vars on startup, logs missing/misconfigured vars.

- [ ] **TEST** — Write test: `diagnoseEnvironment()` returns warnings for missing optional settings and errors for missing required ones.
- [ ] **TEST** — Write test: `diagnoseEnvironment()` returns empty arrays when all required settings are present.
- [ ] **IMPL** — Create `ts/packages/config/src/diagnostics.ts`. Uses the settings registry to find all registered settings and checks their env vars.

---

## Package setup

- [ ] Create `ts/packages/config/package.json` with name `@nodetool/config`
- [ ] Create `ts/packages/config/tsconfig.json`
- [ ] Add to `ts/package.json` workspaces
- [ ] Export from `ts/packages/config/src/index.ts`
