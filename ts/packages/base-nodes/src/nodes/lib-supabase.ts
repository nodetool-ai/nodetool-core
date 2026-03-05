import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

type FilterOp = "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "like" | "contains";
type Filter = [string, FilterOp, unknown];

function getSupabaseClient(url: string, key: string): SupabaseClient {
  if (!url || !key) {
    throw new Error("Supabase URL and key are required. Provide supabase_url and supabase_key.");
  }
  return createClient(url, key);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function applyFilters(query: any, filters: Filter[]): any {
  let q = query;
  for (const [field, op, value] of filters) {
    switch (op) {
      case "eq": q = q.eq(field, value); break;
      case "ne": q = q.neq(field, value); break;
      case "gt": q = q.gt(field, value); break;
      case "gte": q = q.gte(field, value); break;
      case "lt": q = q.lt(field, value); break;
      case "lte": q = q.lte(field, value); break;
      case "in": q = q.in(field, value as unknown[]); break;
      case "like": q = q.like(field, value as string); break;
      case "contains": q = q.contains(field, value as Record<string, unknown>); break;
      default: throw new Error(`Unsupported filter operator: ${op}`);
    }
  }
  return q;
}

export class SelectLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.Select";
  static readonly title = "Select";
  static readonly description = "Query records from a Supabase table.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      table_name: "",
      columns: { type: "record_type", columns: [] },
      filters: [] as Filter[],
      order_by: "",
      descending: false,
      limit: 0,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "");
    const columnsInput = (inputs.columns ?? this._props.columns ?? { columns: [] }) as {
      columns?: Array<{ name: string }>;
    };
    const cols = columnsInput.columns ?? [];
    const filters = (inputs.filters ?? this._props.filters ?? []) as Filter[];
    const orderBy = String(inputs.order_by ?? this._props.order_by ?? "");
    const descending = Boolean(inputs.descending ?? this._props.descending ?? false);
    const limit = Number(inputs.limit ?? this._props.limit ?? 0);

    if (!tableName) throw new Error("table_name cannot be empty");

    const client = getSupabaseClient(url, key);
    const selectColumns = cols.length === 0 ? "*" : cols.map((c) => c.name).join(", ");

    let query = client.from(tableName).select(selectColumns);

    if (filters.length > 0) {
      query = applyFilters(query, filters);
    }
    if (orderBy) {
      query = query.order(orderBy, { ascending: !descending });
    }
    if (limit > 0) {
      query = query.limit(limit);
    }

    const { data, error } = await query;
    if (error) throw new Error(`Supabase select error: ${error.message}`);

    return { output: data ?? [] };
  }
}

export class InsertLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.Insert";
  static readonly title = "Insert";
  static readonly description = "Insert record(s) into a Supabase table.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      table_name: "",
      records: [] as Record<string, unknown>[],
      return_rows: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "");
    const recordsInput = inputs.records ?? this._props.records ?? [];
    const returnRows = Boolean(inputs.return_rows ?? this._props.return_rows ?? true);

    if (!tableName) throw new Error("table_name cannot be empty");

    const data: Record<string, unknown>[] = Array.isArray(recordsInput)
      ? recordsInput as Record<string, unknown>[]
      : [recordsInput as Record<string, unknown>];

    const client = getSupabaseClient(url, key);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let query: any = client.from(tableName).insert(data);
    if (returnRows) {
      query = query.select("*");
    }

    const { data: result, error } = await query;
    if (error) throw new Error(`Supabase insert error: ${error.message}`);

    return returnRows ? { output: result } : { output: { inserted: data.length } };
  }
}

export class UpdateLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.Update";
  static readonly title = "Update";
  static readonly description = "Update records in a Supabase table.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      table_name: "",
      values: {} as Record<string, unknown>,
      filters: [] as Filter[],
      return_rows: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "");
    const values = (inputs.values ?? this._props.values ?? {}) as Record<string, unknown>;
    const filters = (inputs.filters ?? this._props.filters ?? []) as Filter[];
    const returnRows = Boolean(inputs.return_rows ?? this._props.return_rows ?? true);

    if (!tableName) throw new Error("table_name cannot be empty");
    if (Object.keys(values).length === 0) throw new Error("values cannot be empty");

    const client = getSupabaseClient(url, key);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let query: any = client.from(tableName).update(values);

    if (filters.length > 0) {
      query = applyFilters(query, filters);
    }
    if (returnRows) {
      query = query.select("*");
    }

    const { data, error } = await query;
    if (error) throw new Error(`Supabase update error: ${error.message}`);

    return returnRows ? { output: data } : { output: { updated: true } };
  }
}

export class DeleteLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.Delete";
  static readonly title = "Delete";
  static readonly description = "Delete records from a Supabase table.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      table_name: "",
      filters: [] as Filter[],
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "");
    const filters = (inputs.filters ?? this._props.filters ?? []) as Filter[];

    if (!tableName) throw new Error("table_name cannot be empty");
    if (filters.length === 0) {
      throw new Error("At least one filter is required for DELETE operations to prevent accidental data loss");
    }

    const client = getSupabaseClient(url, key);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let query: any = client.from(tableName).delete();
    query = applyFilters(query, filters);

    const { error } = await query;
    if (error) throw new Error(`Supabase delete error: ${error.message}`);

    return { output: { deleted: true } };
  }
}

export class UpsertLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.Upsert";
  static readonly title = "Upsert";
  static readonly description = "Insert or update (upsert) records in a Supabase table.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      table_name: "",
      records: [] as Record<string, unknown>[],
      return_rows: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const tableName = String(inputs.table_name ?? this._props.table_name ?? "");
    const recordsInput = inputs.records ?? this._props.records ?? [];
    const returnRows = Boolean(inputs.return_rows ?? this._props.return_rows ?? true);

    if (!tableName) throw new Error("table_name cannot be empty");

    const data: Record<string, unknown>[] = Array.isArray(recordsInput)
      ? recordsInput as Record<string, unknown>[]
      : [recordsInput as Record<string, unknown>];

    const client = getSupabaseClient(url, key);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let query: any = client.from(tableName).upsert(data);

    if (returnRows) {
      query = query.select("*");
    }

    const { data: result, error } = await query;
    if (error) throw new Error(`Supabase upsert error: ${error.message}`);

    return returnRows ? { output: result } : { output: { upserted: data.length } };
  }
}

export class RPCLibNode extends BaseNode {
  static readonly nodeType = "lib.supabase.RPC";
  static readonly title = "RPC";
  static readonly description = "Call a PostgreSQL function via Supabase RPC.";

  defaults() {
    return {
      supabase_url: "",
      supabase_key: "",
      function: "",
      params: {} as Record<string, unknown>,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.supabase_url ?? this._props.supabase_url ?? "");
    const key = String(inputs.supabase_key ?? this._props.supabase_key ?? "");
    const fnName = String(inputs.function ?? this._props.function ?? "");
    const params = (inputs.params ?? this._props.params ?? {}) as Record<string, unknown>;

    if (!fnName) throw new Error("function cannot be empty");

    const client = getSupabaseClient(url, key);
    const { data, error } = await client.rpc(fnName, params);
    if (error) throw new Error(`Supabase RPC error: ${error.message}`);

    return { output: data };
  }
}

export const LIB_SUPABASE_NODES: readonly NodeClass[] = [
  SelectLibNode as unknown as NodeClass,
  InsertLibNode as unknown as NodeClass,
  UpdateLibNode as unknown as NodeClass,
  DeleteLibNode as unknown as NodeClass,
  UpsertLibNode as unknown as NodeClass,
  RPCLibNode as unknown as NodeClass,
];
