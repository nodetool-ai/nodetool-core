/**
 * OpenAPI-level parity checker.
 *
 * Compares Python FastAPI OpenAPI schema (exported by
 * ``scripts/export_parity_snapshot.py openapi``) against a TypeScript OpenAPI
 * schema (partial — only the routes the TS server has defined).
 *
 * Since the TS Express server does not auto-generate OpenAPI, the TS schema
 * may be a subset of the Python schema. Missing TS routes are warnings, not
 * errors.
 *
 * Response field comparison resolves `$ref` references within the same schema
 * document and maps JSON Schema types to the canonical parity type set:
 * ``"string" | "number" | "boolean" | "json" | "datetime"``
 */

// ── Public types ──────────────────────────────────────────────────────

export interface OpenApiField {
  type: string;
  required: boolean;
}

export interface OpenApiRoute {
  method: string;
  path: string;
  operationId?: string;
  responseFields: Record<string, OpenApiField>;
}

/** key = "METHOD /path", e.g. "GET /api/workflows" */
export type OpenApiRouteMap = Record<string, OpenApiRoute>;

export interface OpenApiDrift {
  route: string;
  field: string;
  severity: "error" | "warning" | "info";
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface OpenApiParityReport {
  pass: boolean;
  drifts: OpenApiDrift[];
  summary: {
    routesChecked: number;
    routesMatched: number;
    fieldsChecked: number;
    errors: number;
    warnings: number;
  };
}

// ── JSON Schema type mapping ──────────────────────────────────────────

type CanonicalType = "string" | "number" | "boolean" | "json" | "datetime";

/**
 * Map a JSON Schema type string (or format hint) to a canonical parity type.
 */
function jsonSchemaTypeToCanonical(
  type: string | undefined,
  format: string | undefined,
): CanonicalType {
  if (format === "date-time" || format === "date") return "datetime";
  switch (type) {
    case "string":
      return "string";
    case "integer":
    case "number":
      return "number";
    case "boolean":
      return "boolean";
    case "array":
    case "object":
      return "json";
    default:
      return "json";
  }
}

/**
 * Resolve a single JSON Schema property object to a canonical type string,
 * following `$ref`, `anyOf`, `oneOf`, `allOf` as needed.
 */
function resolvePropertyType(
  prop: Record<string, unknown>,
  components: Record<string, unknown>,
  depth = 0,
): CanonicalType {
  if (depth > 10) return "json"; // guard infinite recursion

  // Handle $ref
  if ("$ref" in prop && typeof prop["$ref"] === "string") {
    const resolved = resolveRef(prop["$ref"] as string, components);
    if (resolved) return resolvePropertyType(resolved, components, depth + 1);
    return "json";
  }

  // anyOf / oneOf — filter out nulls, recurse into the single non-null type
  for (const key of ["anyOf", "oneOf"]) {
    const variants = prop[key];
    if (Array.isArray(variants)) {
      const nonNull = (variants as Record<string, unknown>[]).filter(
        (v) => !(v["type"] === "null" || ("const" in v && v["const"] === null)),
      );
      if (nonNull.length === 1) {
        return resolvePropertyType(nonNull[0], components, depth + 1);
      }
      return "json";
    }
  }

  // allOf — take first element
  const allOf = prop["allOf"];
  if (Array.isArray(allOf) && allOf.length > 0) {
    return resolvePropertyType(
      allOf[0] as Record<string, unknown>,
      components,
      depth + 1,
    );
  }

  const type = prop["type"] as string | undefined;
  const format = prop["format"] as string | undefined;

  // If it's a string with date-time format, treat as datetime
  return jsonSchemaTypeToCanonical(type, format);
}

// ── $ref resolution ───────────────────────────────────────────────────

/**
 * Resolve a JSON Pointer `$ref` string within the given components map.
 * Only handles `#/components/schemas/<Name>` style references.
 */
function resolveRef(
  ref: string,
  components: Record<string, unknown>,
): Record<string, unknown> | null {
  // e.g. "#/components/schemas/WorkflowList"
  if (!ref.startsWith("#/")) return null;
  const parts = ref.slice(2).split("/");
  // parts[0] = "components", parts[1] = "schemas", parts[2] = name
  if (parts[0] !== "components" || parts[1] !== "schemas") return null;
  const schemaName = parts[2];
  const schemas = components["schemas"];
  if (!schemas || typeof schemas !== "object") return null;
  const schema = (schemas as Record<string, unknown>)[schemaName];
  if (!schema || typeof schema !== "object") return null;
  return schema as Record<string, unknown>;
}

/**
 * Extract the response field map from a resolved schema object.
 * Handles object schemas with `properties` and `required` arrays.
 */
function extractFieldsFromSchema(
  schema: Record<string, unknown>,
  components: Record<string, unknown>,
  requiredSet: Set<string>,
): Record<string, OpenApiField> {
  const fields: Record<string, OpenApiField> = {};

  // Follow $ref at the top level
  if ("$ref" in schema && typeof schema["$ref"] === "string") {
    const resolved = resolveRef(schema["$ref"] as string, components);
    if (resolved) {
      const resolvedRequired = new Set<string>(
        Array.isArray(resolved["required"])
          ? (resolved["required"] as string[])
          : [],
      );
      return extractFieldsFromSchema(resolved, components, resolvedRequired);
    }
    return fields;
  }

  const properties = schema["properties"];
  if (!properties || typeof properties !== "object") return fields;

  for (const [fieldName, propRaw] of Object.entries(
    properties as Record<string, unknown>,
  )) {
    const prop = propRaw as Record<string, unknown>;
    const canonicalType = resolvePropertyType(prop, components);
    fields[fieldName] = {
      type: canonicalType,
      required: requiredSet.has(fieldName),
    };
  }

  return fields;
}

// ── Route extraction ──────────────────────────────────────────────────

/**
 * Parse an OpenAPI 3.x schema object and extract all routes with their
 * 200-response field maps.
 *
 * @param openApiSchema - Raw OpenAPI 3.x schema object
 * @returns Map from "METHOD /path" to OpenApiRoute
 */
export function extractRoutes(
  openApiSchema: Record<string, unknown>,
): OpenApiRouteMap {
  const routeMap: OpenApiRouteMap = {};

  const paths = openApiSchema["paths"];
  if (!paths || typeof paths !== "object") return routeMap;

  const components =
    typeof openApiSchema["components"] === "object" &&
    openApiSchema["components"] !== null
      ? (openApiSchema["components"] as Record<string, unknown>)
      : {};

  const httpMethods = [
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
  ];

  for (const [pathStr, pathItemRaw] of Object.entries(
    paths as Record<string, unknown>,
  )) {
    if (!pathItemRaw || typeof pathItemRaw !== "object") continue;
    const pathItem = pathItemRaw as Record<string, unknown>;

    for (const method of httpMethods) {
      const operationRaw = pathItem[method];
      if (!operationRaw || typeof operationRaw !== "object") continue;
      const operation = operationRaw as Record<string, unknown>;

      const operationId = operation["operationId"] as string | undefined;
      const responseFields = extract200ResponseFields(operation, components);

      const key = `${method.toUpperCase()} ${pathStr}`;
      routeMap[key] = {
        method: method.toUpperCase(),
        path: pathStr,
        operationId,
        responseFields,
      };
    }
  }

  return routeMap;
}

/**
 * Extract field definitions from a route operation's 200 response schema.
 */
function extract200ResponseFields(
  operation: Record<string, unknown>,
  components: Record<string, unknown>,
): Record<string, OpenApiField> {
  const responses = operation["responses"];
  if (!responses || typeof responses !== "object") return {};

  const responsesObj = responses as Record<string, unknown>;
  const response200Raw = responsesObj["200"] ?? responsesObj[200];
  if (!response200Raw || typeof response200Raw !== "object") return {};
  const response200 = response200Raw as Record<string, unknown>;

  const content = response200["content"];
  if (!content || typeof content !== "object") return {};
  const contentObj = content as Record<string, unknown>;

  // Try application/json first, then any media type
  const mediaTypeRaw =
    contentObj["application/json"] ??
    Object.values(contentObj).find((v) => v && typeof v === "object");
  if (!mediaTypeRaw || typeof mediaTypeRaw !== "object") return {};
  const mediaType = mediaTypeRaw as Record<string, unknown>;

  const schemaRaw = mediaType["schema"];
  if (!schemaRaw || typeof schemaRaw !== "object") return {};
  const schema = schemaRaw as Record<string, unknown>;

  // Determine required fields at the response schema level
  let requiredSet = new Set<string>();

  // If schema is a $ref, resolve first to get required
  if ("$ref" in schema && typeof schema["$ref"] === "string") {
    const resolved = resolveRef(schema["$ref"] as string, components);
    if (resolved) {
      requiredSet = new Set<string>(
        Array.isArray(resolved["required"])
          ? (resolved["required"] as string[])
          : [],
      );
    }
  } else {
    requiredSet = new Set<string>(
      Array.isArray(schema["required"]) ? (schema["required"] as string[]) : [],
    );
  }

  return extractFieldsFromSchema(schema, components, requiredSet);
}

// ── Comparison logic ──────────────────────────────────────────────────

/**
 * Compare two OpenAPI schemas — one from Python (authoritative) and one from
 * TypeScript (may be partial / absent for routes not yet ported).
 *
 * Severity rules:
 * - `warning`: A Python route is not present in TS (TS may intentionally omit)
 * - `error`: A required Python response field is missing in the TS 200 response
 * - `warning`: A shared field has a type mismatch
 * - `info`: TS has extra fields not present in Python
 */
export function compareOpenApiSchemas(
  pythonSchema: Record<string, unknown>,
  tsSchema: Record<string, unknown>,
): OpenApiParityReport {
  const drifts: OpenApiDrift[] = [];

  const pyRoutes = extractRoutes(pythonSchema);
  const tsRoutes = extractRoutes(tsSchema);

  let routesMatched = 0;
  let fieldsChecked = 0;

  for (const [routeKey, pyRoute] of Object.entries(pyRoutes)) {
    const tsRoute = tsRoutes[routeKey];

    if (!tsRoute) {
      drifts.push({
        route: routeKey,
        field: "(route)",
        severity: "warning",
        message: `Route ${routeKey} exists in Python but not in TypeScript`,
        python: { operationId: pyRoute.operationId },
      });
      continue;
    }

    routesMatched++;

    // Compare response fields
    const pyFields = pyRoute.responseFields;
    const tsFields = tsRoute.responseFields;

    for (const [fieldName, pyField] of Object.entries(pyFields)) {
      fieldsChecked++;
      const tsField = tsFields[fieldName];

      if (!tsField) {
        if (pyField.required) {
          drifts.push({
            route: routeKey,
            field: fieldName,
            severity: "error",
            message: `Required field "${fieldName}" in Python 200 response is missing in TypeScript`,
            python: pyField,
          });
        } else {
          drifts.push({
            route: routeKey,
            field: fieldName,
            severity: "warning",
            message: `Optional field "${fieldName}" in Python 200 response is missing in TypeScript`,
            python: pyField,
          });
        }
        continue;
      }

      // Type mismatch
      if (pyField.type !== tsField.type) {
        drifts.push({
          route: routeKey,
          field: fieldName,
          severity: "warning",
          message: `Type mismatch for field "${fieldName}"`,
          python: pyField.type,
          ts: tsField.type,
        });
      }
    }

    // Extra fields in TS not in Python
    for (const fieldName of Object.keys(tsFields)) {
      if (!(fieldName in pyFields)) {
        drifts.push({
          route: routeKey,
          field: fieldName,
          severity: "info",
          message: `Field "${fieldName}" exists in TypeScript 200 response but not in Python`,
          ts: tsFields[fieldName],
        });
      }
    }
  }

  // Routes in TS but not Python — info only
  for (const routeKey of Object.keys(tsRoutes)) {
    if (!(routeKey in pyRoutes)) {
      drifts.push({
        route: routeKey,
        field: "(route)",
        severity: "info",
        message: `Route ${routeKey} exists in TypeScript but not in Python`,
        ts: { operationId: tsRoutes[routeKey].operationId },
      });
    }
  }

  const errors = drifts.filter((d) => d.severity === "error").length;
  const warnings = drifts.filter((d) => d.severity === "warning").length;

  return {
    pass: errors === 0,
    drifts,
    summary: {
      routesChecked: Object.keys(pyRoutes).length,
      routesMatched,
      fieldsChecked,
      errors,
      warnings,
    },
  };
}
