export {
  diffMessageStreams,
  type Drift,
  type DriftCategory,
  type DiffOptions,
} from "./diff.js";

export {
  runShadowComparison,
  type CommandSpec,
  type ShadowRunResult,
  type ShadowComparison,
} from "./shadow.js";

export {
  evaluateCanaryGates,
  DEFAULT_THRESHOLDS,
  type CanaryMetrics,
  type CanaryThresholds,
  type CanaryGateResult,
} from "./gates.js";

export {
  compareModelSchemas,
  type ColumnDef,
  type IndexDef as ModelIndexDef,
  type ModelSchema,
  type ModelSchemaMap,
  type ModelDrift,
  type ModelParityReport,
} from "./model-parity.js";

export {
  compareApiRoutes,
  type ApiRoute,
  type ApiDrift,
  type ApiParityReport,
} from "./api-parity.js";

export {
  compareCliCommands,
  type CliCommand,
  type CliParam,
  type CliDrift,
  type CliParityReport,
} from "./cli-parity.js";

export {
  compareLibraryClasses,
  snakeToCamel,
  type LibraryParam,
  type LibraryMethod,
  type LibraryClass,
  type LibraryDrift,
  type LibraryParityReport,
  type TsMethodDef,
  type TsClassDef,
} from "./library-parity.js";

export {
  extractRoutes,
  compareOpenApiSchemas,
  type OpenApiField,
  type OpenApiRoute,
  type OpenApiRouteMap,
  type OpenApiDrift,
  type OpenApiParityReport,
} from "./openapi-parity.js";

export {
  compareProtocolMessages,
  TS_MESSAGE_MANIFEST,
  type MessageFieldDef,
  type MessageSchema,
  type MessageSchemaMap,
  type ProtocolDrift,
  type ProtocolParityReport,
} from "./protocol-parity.js";

export {
  fetchJson,
  compareHttpResponses,
  type HttpShadowOptions,
  type HttpShadowResult,
} from "./shadow.js";
