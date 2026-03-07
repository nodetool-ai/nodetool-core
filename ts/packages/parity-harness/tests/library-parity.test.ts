import { describe, expect, it } from "vitest";
import {
  compareLibraryClasses,
  snakeToCamel,
  type LibraryClass,
  type TsClassDef,
} from "../src/library-parity.js";

describe("snakeToCamel", () => {
  it("converts snake_case to camelCase", () => {
    expect(snakeToCamel("find_node")).toBe("findNode");
    expect(snakeToCamel("get_table_schema")).toBe("getTableSchema");
    expect(snakeToCamel("validate")).toBe("validate");
    expect(snakeToCamel("a_b_c")).toBe("aBC");
  });
});

describe("compareLibraryClasses", () => {
  it("returns pass=true when all methods match", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "models",
        class: "DBModel",
        methods: [
          { name: "save", params: [] },
          { name: "delete", params: [] },
          { name: "get_etag", params: [] },
        ],
      },
    ];
    const tsClasses: TsClassDef[] = [
      {
        name: "DBModel",
        methods: [
          { name: "save", paramCount: 0 },
          { name: "delete", paramCount: 0 },
          { name: "getEtag", paramCount: 0 },
        ],
      },
    ];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.pass).toBe(true);
    expect(report.summary.matched).toBe(3);
  });

  it("detects methods in Python but not TS", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "models",
        class: "DBModel",
        methods: [
          { name: "save", params: [] },
          { name: "special_method", params: [] },
        ],
      },
    ];
    const tsClasses: TsClassDef[] = [
      {
        name: "DBModel",
        methods: [{ name: "save", paramCount: 0 }],
      },
    ];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.summary.pythonOnly).toBe(1);
    expect(report.drifts.some((d) => d.method === "special_method")).toBe(true);
  });

  it("detects methods in TS but not Python", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "models",
        class: "DBModel",
        methods: [{ name: "save", params: [] }],
      },
    ];
    const tsClasses: TsClassDef[] = [
      {
        name: "DBModel",
        methods: [
          { name: "save", paramCount: 0 },
          { name: "toRow", paramCount: 0 },
        ],
      },
    ];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.summary.tsOnly).toBe(1);
  });

  it("handles missing TS class", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "models",
        class: "Missing",
        methods: [{ name: "run", params: [] }],
      },
    ];
    const tsClasses: TsClassDef[] = [];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].message).toContain("no TypeScript counterpart");
  });

  it("handles Python introspection errors", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "broken",
        class: "Broken",
        methods: [],
        error: "import failed",
      },
    ];
    const tsClasses: TsClassDef[] = [];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].severity).toBe("warning");
  });

  it("matches snake_case Python methods to camelCase TS methods", () => {
    const pyClasses: LibraryClass[] = [
      {
        module: "graph",
        class: "Graph",
        methods: [
          { name: "find_node", params: [{ name: "node_id", kind: "POSITIONAL_OR_KEYWORD", has_default: false }] },
          { name: "topological_sort", params: [] },
        ],
      },
    ];
    const tsClasses: TsClassDef[] = [
      {
        name: "Graph",
        methods: [
          { name: "findNode", paramCount: 1 },
          { name: "topologicalSort", paramCount: 0 },
        ],
      },
    ];

    const report = compareLibraryClasses(pyClasses, tsClasses);
    expect(report.summary.matched).toBe(2);
    expect(report.summary.pythonOnly).toBe(0);
  });
});
