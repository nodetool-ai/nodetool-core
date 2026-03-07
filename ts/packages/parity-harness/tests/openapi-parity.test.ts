import { describe, expect, it } from "vitest";
import {
  compareOpenApiSchemas,
  extractRoutes,
  type OpenApiDrift,
  type OpenApiField,
  type OpenApiParityReport,
  type OpenApiRoute,
  type OpenApiRouteMap,
} from "../src/openapi-parity.js";

// ── Helpers ───────────────────────────────────────────────────────────

/** Build a minimal OpenAPI 3.x schema with one route and optional 200 response. */
function makeSchema(
  routes: Array<{
    method: string;
    path: string;
    operationId?: string;
    responseFields?: Record<string, { type: string; required?: boolean }>;
  }>,
  extraComponents?: Record<string, unknown>,
): Record<string, unknown> {
  const paths: Record<string, unknown> = {};

  for (const route of routes) {
    const method = route.method.toLowerCase();
    const props: Record<string, unknown> = {};
    const required: string[] = [];

    for (const [fieldName, fieldDef] of Object.entries(
      route.responseFields ?? {},
    )) {
      props[fieldName] = { type: fieldDef.type };
      if (fieldDef.required) required.push(fieldName);
    }

    const responseSchema: Record<string, unknown> =
      Object.keys(props).length > 0
        ? { type: "object", properties: props, required }
        : {};

    const operation: Record<string, unknown> = {
      operationId: route.operationId ?? `${method}_${route.path.replace(/\//g, "_")}`,
      responses: {
        "200": {
          description: "OK",
          content: {
            "application/json": {
              schema: responseSchema,
            },
          },
        },
      },
    };

    if (!paths[route.path]) {
      paths[route.path] = {};
    }
    (paths[route.path] as Record<string, unknown>)[method] = operation;
  }

  return {
    openapi: "3.1.0",
    info: { title: "Test", version: "0.0.1" },
    paths,
    components: {
      schemas: extraComponents ?? {},
    },
  };
}

// ── extractRoutes tests ───────────────────────────────────────────────

describe("extractRoutes", () => {
  it("returns empty map for schema with no paths", () => {
    const schema = { openapi: "3.1.0", paths: {}, components: { schemas: {} } };
    const routes = extractRoutes(schema);
    expect(Object.keys(routes)).toHaveLength(0);
  });

  it("parses a single GET route", () => {
    const schema = makeSchema([
      { method: "GET", path: "/api/items", operationId: "list_items" },
    ]);
    const routes = extractRoutes(schema);
    expect(routes["GET /api/items"]).toBeDefined();
    expect(routes["GET /api/items"].method).toBe("GET");
    expect(routes["GET /api/items"].path).toBe("/api/items");
    expect(routes["GET /api/items"].operationId).toBe("list_items");
  });

  it("parses multiple methods on the same path", () => {
    const schema = makeSchema([
      { method: "GET", path: "/api/items" },
      { method: "POST", path: "/api/items" },
    ]);
    const routes = extractRoutes(schema);
    expect(routes["GET /api/items"]).toBeDefined();
    expect(routes["POST /api/items"]).toBeDefined();
  });

  it("parses multiple paths", () => {
    const schema = makeSchema([
      { method: "GET", path: "/api/items" },
      { method: "DELETE", path: "/api/items/{id}" },
      { method: "PUT", path: "/api/users/{id}" },
    ]);
    const routes = extractRoutes(schema);
    expect(Object.keys(routes)).toHaveLength(3);
  });

  it("extracts response fields from inline properties", () => {
    const schema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          name: { type: "string", required: true },
          count: { type: "integer", required: false },
        },
      },
    ]);
    const routes = extractRoutes(schema);
    const route = routes["GET /api/items"];
    expect(route).toBeDefined();
    expect(route.responseFields["id"]).toEqual({ type: "string", required: true });
    expect(route.responseFields["name"]).toEqual({ type: "string", required: true });
    expect(route.responseFields["count"]).toEqual({ type: "number", required: false });
  });

  it("maps JSON Schema types to canonical parity types", () => {
    const schema: Record<string, unknown> = {
      openapi: "3.1.0",
      paths: {
        "/api/test": {
          get: {
            operationId: "test",
            responses: {
              "200": {
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      required: ["s", "n", "b", "a", "o"],
                      properties: {
                        s: { type: "string" },
                        n: { type: "integer" },
                        f: { type: "number" },
                        b: { type: "boolean" },
                        a: { type: "array" },
                        o: { type: "object" },
                        dt: { type: "string", format: "date-time" },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      components: { schemas: {} },
    };

    const routes = extractRoutes(schema);
    const fields = routes["GET /api/test"].responseFields;
    expect(fields["s"].type).toBe("string");
    expect(fields["n"].type).toBe("number");
    expect(fields["f"].type).toBe("number");
    expect(fields["b"].type).toBe("boolean");
    expect(fields["a"].type).toBe("json");
    expect(fields["o"].type).toBe("json");
    expect(fields["dt"].type).toBe("datetime");
  });

  it("resolves $ref in 200 response schema", () => {
    const schema: Record<string, unknown> = {
      openapi: "3.1.0",
      paths: {
        "/api/items": {
          get: {
            operationId: "list_items",
            responses: {
              "200": {
                content: {
                  "application/json": {
                    schema: { $ref: "#/components/schemas/ItemList" },
                  },
                },
              },
            },
          },
        },
      },
      components: {
        schemas: {
          ItemList: {
            type: "object",
            required: ["items"],
            properties: {
              items: { type: "array" },
              total: { type: "integer" },
            },
          },
        },
      },
    };

    const routes = extractRoutes(schema);
    const route = routes["GET /api/items"];
    expect(route).toBeDefined();
    expect(route.responseFields["items"]).toEqual({ type: "json", required: true });
    expect(route.responseFields["total"]).toEqual({ type: "number", required: false });
  });

  it("resolves anyOf with nullable types", () => {
    const schema: Record<string, unknown> = {
      openapi: "3.1.0",
      paths: {
        "/api/test": {
          get: {
            operationId: "test",
            responses: {
              "200": {
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: {
                        name: {
                          anyOf: [{ type: "string" }, { type: "null" }],
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      components: { schemas: {} },
    };

    const routes = extractRoutes(schema);
    expect(routes["GET /api/test"].responseFields["name"].type).toBe("string");
  });

  it("returns empty responseFields when no 200 response is defined", () => {
    const schema: Record<string, unknown> = {
      openapi: "3.1.0",
      paths: {
        "/api/items": {
          delete: {
            operationId: "delete_item",
            responses: {
              "204": { description: "No Content" },
            },
          },
        },
      },
      components: { schemas: {} },
    };

    const routes = extractRoutes(schema);
    expect(routes["DELETE /api/items"]).toBeDefined();
    expect(Object.keys(routes["DELETE /api/items"].responseFields)).toHaveLength(0);
  });
});

// ── compareOpenApiSchemas tests ───────────────────────────────────────

describe("compareOpenApiSchemas", () => {
  it("returns pass=true and zero drifts for identical schemas", () => {
    const schema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        operationId: "list_items",
        responseFields: {
          id: { type: "string", required: true },
          name: { type: "string", required: true },
        },
      },
    ]);

    const report = compareOpenApiSchemas(schema, schema);
    expect(report.pass).toBe(true);
    expect(report.drifts).toHaveLength(0);
    expect(report.summary.errors).toBe(0);
    expect(report.summary.warnings).toBe(0);
    expect(report.summary.routesChecked).toBe(1);
    expect(report.summary.routesMatched).toBe(1);
  });

  it("produces warning drift when Python route is missing in TS", () => {
    const pySchema = makeSchema([
      { method: "GET", path: "/api/items" },
      { method: "DELETE", path: "/api/items/{id}" },
    ]);
    const tsSchema = makeSchema([{ method: "GET", path: "/api/items" }]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    expect(report.pass).toBe(true); // warnings don't fail
    const missingRoute = report.drifts.find(
      (d) => d.route === "DELETE /api/items/{id}" && d.severity === "warning",
    );
    expect(missingRoute).toBeDefined();
    expect(missingRoute!.message).toContain("exists in Python but not in TypeScript");
  });

  it("produces error drift when a required Python response field is missing in TS", () => {
    const pySchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          name: { type: "string", required: true },
        },
      },
    ]);
    const tsSchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          // name missing
        },
      },
    ]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    expect(report.pass).toBe(false);
    expect(report.summary.errors).toBeGreaterThan(0);

    const errorDrift = report.drifts.find(
      (d) => d.field === "name" && d.severity === "error",
    );
    expect(errorDrift).toBeDefined();
    expect(errorDrift!.message).toContain("Required field");
    expect(errorDrift!.message).toContain("missing in TypeScript");
  });

  it("produces warning drift when an optional Python field is missing in TS", () => {
    const pySchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          description: { type: "string", required: false },
        },
      },
    ]);
    const tsSchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          // description missing (optional in Python)
        },
      },
    ]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    expect(report.pass).toBe(true); // only a warning
    const warningDrift = report.drifts.find(
      (d) => d.field === "description" && d.severity === "warning",
    );
    expect(warningDrift).toBeDefined();
  });

  it("produces info drifts for extra TS fields not in Python", () => {
    const pySchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
        },
      },
    ]);
    const tsSchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          extra_ts_field: { type: "string", required: false },
        },
      },
    ]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    expect(report.pass).toBe(true);
    const infoDrift = report.drifts.find(
      (d) => d.field === "extra_ts_field" && d.severity === "info",
    );
    expect(infoDrift).toBeDefined();
    expect(infoDrift!.message).toContain("exists in TypeScript");
    expect(infoDrift!.message).toContain("not in Python");
  });

  it("produces warning drift for type mismatches on shared fields", () => {
    const pySchema = makeSchema([
      {
        method: "GET",
        path: "/api/items",
        responseFields: {
          id: { type: "string", required: true },
          count: { type: "integer", required: true },
        },
      },
    ]);

    // Build a TS schema where count has a different type
    const tsSchema: Record<string, unknown> = {
      openapi: "3.1.0",
      paths: {
        "/api/items": {
          get: {
            operationId: "list_items",
            responses: {
              "200": {
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      required: ["id", "count"],
                      properties: {
                        id: { type: "string" },
                        count: { type: "string" }, // mismatch: string instead of integer/number
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
      components: { schemas: {} },
    };

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    const typeDrift = report.drifts.find(
      (d) => d.field === "count" && d.severity === "warning",
    );
    expect(typeDrift).toBeDefined();
    expect(typeDrift!.message).toContain("Type mismatch");
    expect(typeDrift!.python).toBe("number");
    expect(typeDrift!.ts).toBe("string");
  });

  it("produces info drift for TS-only routes", () => {
    const pySchema = makeSchema([]);
    const tsSchema = makeSchema([
      { method: "GET", path: "/api/ts-only" },
    ]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    const infoDrift = report.drifts.find(
      (d) => d.route === "GET /api/ts-only" && d.severity === "info",
    );
    expect(infoDrift).toBeDefined();
  });

  it("summary counts are accurate", () => {
    const pySchema = makeSchema([
      {
        method: "GET",
        path: "/api/a",
        responseFields: {
          x: { type: "string", required: true },
          y: { type: "string", required: false },
        },
      },
      { method: "POST", path: "/api/b" },
    ]);
    const tsSchema = makeSchema([
      {
        method: "GET",
        path: "/api/a",
        responseFields: {
          x: { type: "string", required: true },
          y: { type: "string", required: false },
        },
      },
      // /api/b missing in TS — produces warning
    ]);

    const report = compareOpenApiSchemas(pySchema, tsSchema);
    expect(report.summary.routesChecked).toBe(2);
    expect(report.summary.routesMatched).toBe(1);
    expect(report.summary.fieldsChecked).toBe(2); // x + y from /api/a
  });

  it("handles schemas with no paths gracefully", () => {
    const empty = { openapi: "3.1.0", paths: {}, components: { schemas: {} } };
    const report = compareOpenApiSchemas(empty, empty);
    expect(report.pass).toBe(true);
    expect(report.drifts).toHaveLength(0);
  });
});
