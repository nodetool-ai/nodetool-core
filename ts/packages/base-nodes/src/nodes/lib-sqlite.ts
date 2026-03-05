import Database from "better-sqlite3";
import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";
import { join } from "node:path";
import { existsSync } from "node:fs";

function quoteIdentifier(name: string): string {
  const escaped = name.replace(/"/g, '""');
  return `"${escaped}"`;
}

function columnTypeToSqlite(columnType: string): string {
  const mapping: Record<string, string> = {
    int: "INTEGER",
    float: "REAL",
    datetime: "TEXT",
    string: "TEXT",
    object: "TEXT",
  };
  return mapping[columnType] ?? "TEXT";
}

function resolveDbPath(context: ProcessingContext | undefined, databaseName: string): string {
  const workspaceDir = context?.workspaceDir;
  if (!workspaceDir) {
    throw new Error("workspace_dir is required for SQLite operations");
  }
  return join(workspaceDir, databaseName);
}

function serializeValue(v: unknown): unknown {
  if (v !== null && typeof v === "object") {
    return JSON.stringify(v);
  }
  return v;
}

function tryParseJsonValues(row: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (typeof value === "string") {
      try {
        result[key] = JSON.parse(value);
      } catch {
        result[key] = value;
      }
    } else {
      result[key] = value;
    }
  }
  return result;
}

export class CreateTableLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.CreateTable";
  static readonly title = "Create Table";
  static readonly description = "Create a new SQLite table with specified columns.";

  defaults() {
    return {
      database_name: "memory.db",
      table_name: "flashcards",
      columns: { type: "record_type", columns: [] },
      add_primary_key: true,
      if_not_exists: true,
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "flashcards");
    const columnsInput = (inputs.columns ?? this._props.columns ?? { columns: [] }) as {
      columns?: Array<{ name: string; data_type: string }>;
    };
    const columns = columnsInput.columns ?? [];
    const addPrimaryKey = inputs.add_primary_key ?? this._props.add_primary_key ?? true;
    const ifNotExists = inputs.if_not_exists ?? this._props.if_not_exists ?? true;

    const dbPath = resolveDbPath(context, databaseName);
    const db = new Database(dbPath);

    try {
      const existing = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .get(tableName) as Record<string, unknown> | undefined;

      if (existing) {
        return { database_name: databaseName, table_name: tableName, columns: columnsInput };
      }

      const columnDefs: string[] = [];
      for (let i = 0; i < columns.length; i++) {
        const col = columns[i];
        const sqliteType = columnTypeToSqlite(col.data_type);
        const colName = quoteIdentifier(col.name);

        if (i === 0 && addPrimaryKey && col.data_type === "int") {
          columnDefs.push(`${colName} INTEGER PRIMARY KEY AUTOINCREMENT`);
        } else {
          columnDefs.push(`${colName} ${sqliteType}`);
        }
      }

      const columnsSql = columnDefs.join(", ");
      const ifNotExistsClause = ifNotExists ? "IF NOT EXISTS " : "";
      const quotedTableName = quoteIdentifier(tableName);
      const sql = `CREATE TABLE ${ifNotExistsClause}${quotedTableName} (${columnsSql})`;

      db.exec(sql);

      return { database_name: databaseName, table_name: tableName, columns: columnsInput };
    } finally {
      db.close();
    }
  }
}

export class InsertLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.Insert";
  static readonly title = "Insert";
  static readonly description = "Insert a record into a SQLite table.";

  defaults() {
    return {
      database_name: "memory.db",
      table_name: "flashcards",
      data: { content: "example" },
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "flashcards");
    const data = (inputs.data ?? this._props.data ?? {}) as Record<string, unknown>;

    const dbPath = resolveDbPath(context, databaseName);
    const db = new Database(dbPath);

    try {
      const keys = Object.keys(data);
      const columnsPart = keys.map((k) => quoteIdentifier(k)).join(", ");
      const placeholders = keys.map(() => "?").join(", ");
      const quotedTableName = quoteIdentifier(tableName);
      const sql = `INSERT INTO ${quotedTableName} (${columnsPart}) VALUES (${placeholders})`;

      const values = Object.values(data).map(serializeValue);
      const result = db.prepare(sql).run(...values);

      return {
        row_id: result.lastInsertRowid,
        rows_affected: result.changes,
        message: `Inserted record with ID ${result.lastInsertRowid}`,
      };
    } finally {
      db.close();
    }
  }
}

export class QueryLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.Query";
  static readonly title = "Query";
  static readonly description = "Query records from a SQLite table.";

  defaults() {
    return {
      database_name: "memory.db",
      table_name: "flashcards",
      where: "",
      columns: { type: "record_type", columns: [] },
      order_by: "",
      limit: 0,
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "flashcards");
    const where = String(inputs.where ?? this._props.where ?? "");
    const columnsInput = (inputs.columns ?? this._props.columns ?? { columns: [] }) as {
      columns?: Array<{ name: string }>;
    };
    const cols = columnsInput.columns ?? [];
    const orderBy = String(inputs.order_by ?? this._props.order_by ?? "");
    const limit = Number(inputs.limit ?? this._props.limit ?? 0);

    const dbPath = resolveDbPath(context, databaseName);

    if (!existsSync(dbPath)) {
      return { output: [] };
    }

    const db = new Database(dbPath);

    try {
      const selectColumns =
        cols.length === 0 ? "*" : cols.map((c) => quoteIdentifier(c.name)).join(", ");

      const quotedTableName = quoteIdentifier(tableName);
      let sql = `SELECT ${selectColumns} FROM ${quotedTableName}`;

      if (where) {
        sql += ` WHERE ${where}`;
      }
      if (orderBy) {
        sql += ` ORDER BY ${orderBy}`;
      }
      if (limit > 0) {
        sql += ` LIMIT ${limit}`;
      }

      const rows = db.prepare(sql).all() as Record<string, unknown>[];
      const results = rows.map(tryParseJsonValues);

      return { output: results };
    } finally {
      db.close();
    }
  }
}

export class UpdateLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.Update";
  static readonly title = "Update";
  static readonly description = "Update records in a SQLite table.";

  defaults() {
    return {
      database_name: "memory.db",
      table_name: "flashcards",
      data: { content: "updated" },
      where: "",
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "flashcards");
    const data = (inputs.data ?? this._props.data ?? {}) as Record<string, unknown>;
    const where = String(inputs.where ?? this._props.where ?? "");

    const dbPath = resolveDbPath(context, databaseName);
    const db = new Database(dbPath);

    try {
      const keys = Object.keys(data);
      const setClause = keys.map((col) => `${quoteIdentifier(col)} = ?`).join(", ");
      const quotedTableName = quoteIdentifier(tableName);
      let sql = `UPDATE ${quotedTableName} SET ${setClause}`;

      if (where) {
        sql += ` WHERE ${where}`;
      }

      const values = Object.values(data).map(serializeValue);
      const result = db.prepare(sql).run(...values);

      return {
        rows_affected: result.changes,
        message: `Updated ${result.changes} record(s)`,
      };
    } finally {
      db.close();
    }
  }
}

export class DeleteLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.Delete";
  static readonly title = "Delete";
  static readonly description = "Delete records from a SQLite table.";

  defaults() {
    return {
      database_name: "memory.db",
      table_name: "flashcards",
      where: "",
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "flashcards");
    const where = String(inputs.where ?? this._props.where ?? "");

    if (!where) {
      throw new Error("WHERE clause is required for DELETE operations to prevent accidental data loss");
    }

    const dbPath = resolveDbPath(context, databaseName);
    const db = new Database(dbPath);

    try {
      const quotedTableName = quoteIdentifier(tableName);
      const sql = `DELETE FROM ${quotedTableName} WHERE ${where}`;

      const result = db.prepare(sql).run();

      return {
        rows_affected: result.changes,
        message: `Deleted ${result.changes} record(s)`,
      };
    } finally {
      db.close();
    }
  }
}

export class ExecuteSQLLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.ExecuteSQL";
  static readonly title = "Execute SQL";
  static readonly description = "Execute arbitrary SQL statements for advanced operations.";

  defaults() {
    return {
      database_name: "memory.db",
      sql: "SELECT * FROM flashcards",
      parameters: [] as unknown[],
    };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const sqlStr = String(inputs.sql ?? this._props.sql ?? "");
    const parameters = (inputs.parameters ?? this._props.parameters ?? []) as unknown[];

    const dbPath = resolveDbPath(context, databaseName);
    const db = new Database(dbPath);

    try {
      const trimmed = sqlStr.trim().toUpperCase();
      const isModifying = /^(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b/.test(trimmed);

      if (isModifying) {
        const result = db.prepare(sqlStr).run(...parameters);
        return {
          rows_affected: result.changes,
          last_row_id: result.lastInsertRowid,
          message: "SQL executed successfully",
        };
      } else {
        const rows = db.prepare(sqlStr).all(...parameters) as Record<string, unknown>[];
        const results = rows.map(tryParseJsonValues);
        return {
          rows: results,
          count: results.length,
          message: `Query returned ${results.length} row(s)`,
        };
      }
    } finally {
      db.close();
    }
  }
}

export class GetDatabasePathLibNode extends BaseNode {
  static readonly nodeType = "lib.sqlite.GetDatabasePath";
  static readonly title = "Get Database Path";
  static readonly description = "Get the full path to a SQLite database file.";

  defaults() {
    return { database_name: "memory.db" };
  }

  async process(inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const databaseName = String(inputs.database_name ?? this._props.database_name ?? "memory.db");
    const dbPath = resolveDbPath(context, databaseName);
    return { output: dbPath };
  }
}

export const LIB_SQLITE_NODES: readonly NodeClass[] = [
  CreateTableLibNode as unknown as NodeClass,
  InsertLibNode as unknown as NodeClass,
  QueryLibNode as unknown as NodeClass,
  UpdateLibNode as unknown as NodeClass,
  DeleteLibNode as unknown as NodeClass,
  ExecuteSQLLibNode as unknown as NodeClass,
  GetDatabasePathLibNode as unknown as NodeClass,
];
