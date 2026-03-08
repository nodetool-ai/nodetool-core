import { describe, expect, it } from "vitest";
import { compareApiRoutes, type ApiRoute } from "../src/api-parity.js";

describe("compareApiRoutes", () => {
  it("returns pass=true when routes are identical", () => {
    const routes: ApiRoute[] = [
      { method: "GET", path: "/api/items", name: "list_items" },
      { method: "POST", path: "/api/items", name: "create_item" },
    ];

    const report = compareApiRoutes(routes, [...routes]);
    expect(report.pass).toBe(true);
    expect(report.drifts).toHaveLength(0);
    expect(report.summary.matched).toBe(2);
  });

  it("detects routes in Python but not TS", () => {
    const pyRoutes: ApiRoute[] = [
      { method: "GET", path: "/api/items", name: "list" },
      { method: "DELETE", path: "/api/items/{id}", name: "delete" },
    ];
    const tsRoutes: ApiRoute[] = [
      { method: "GET", path: "/api/items", name: "list" },
    ];

    const report = compareApiRoutes(pyRoutes, tsRoutes);
    expect(report.summary.pythonOnly).toBe(1);
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].route).toBe("DELETE /api/items/{id}");
  });

  it("detects routes in TS but not Python", () => {
    const pyRoutes: ApiRoute[] = [];
    const tsRoutes: ApiRoute[] = [
      { method: "GET", path: "/api/extra", name: "extra" },
    ];

    const report = compareApiRoutes(pyRoutes, tsRoutes);
    expect(report.summary.tsOnly).toBe(1);
    expect(report.drifts[0].severity).toBe("info");
  });

  it("matches routes by method + path combination", () => {
    const pyRoutes: ApiRoute[] = [
      { method: "GET", path: "/api/items", name: "list" },
      { method: "POST", path: "/api/items", name: "create" },
    ];
    const tsRoutes: ApiRoute[] = [
      { method: "POST", path: "/api/items", name: "create_item" },
      { method: "GET", path: "/api/items", name: "list_items" },
    ];

    const report = compareApiRoutes(pyRoutes, tsRoutes);
    expect(report.summary.matched).toBe(2);
    expect(report.drifts).toHaveLength(0);
  });
});
