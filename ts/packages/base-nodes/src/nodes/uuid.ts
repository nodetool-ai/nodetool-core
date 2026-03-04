import { createHash, randomBytes } from "node:crypto";
import { BaseNode } from "@nodetool/node-sdk";

type UUIDFormat = "standard" | "hex" | "urn" | "int" | "bytes_hex";

const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const UUID_EPOCH_OFFSET = 0x01b21dd213814000n;

const NAMESPACE_UUIDS = {
  dns: "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  url: "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
  oid: "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
  x500: "6ba7b814-9dad-11d1-80b4-00c04fd430c8",
} as const;

let _nodeId: Uint8Array | null = null;
let _clockseq = randomBytes(2).readUInt16BE(0) & 0x3fff;
let _lastMSecs = 0;
let _lastNSecs = 0;

function normalizeUuid(value: string): string {
  let normalized = value.trim().toLowerCase();
  if (normalized.startsWith("urn:uuid:")) {
    normalized = normalized.slice("urn:uuid:".length);
  }
  if (normalized.startsWith("{") && normalized.endsWith("}")) {
    normalized = normalized.slice(1, -1);
  }
  if (/^[0-9a-f]{32}$/i.test(normalized)) {
    normalized = [
      normalized.slice(0, 8),
      normalized.slice(8, 12),
      normalized.slice(12, 16),
      normalized.slice(16, 20),
      normalized.slice(20),
    ].join("-");
  }
  if (!UUID_REGEX.test(normalized)) {
    throw new Error(`Invalid UUID string: ${value}`);
  }
  return normalized;
}

function uuidToBytes(uuid: string): Uint8Array {
  const hex = normalizeUuid(uuid).replaceAll("-", "");
  return Uint8Array.from(Buffer.from(hex, "hex"));
}

function bytesToUuid(bytes: Uint8Array): string {
  const hex = Buffer.from(bytes).toString("hex");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}

function resolveNamespace(namespace: string): string {
  const lower = namespace.toLowerCase();
  if (lower in NAMESPACE_UUIDS) {
    return NAMESPACE_UUIDS[lower as keyof typeof NAMESPACE_UUIDS];
  }
  try {
    return normalizeUuid(namespace);
  } catch {
    throw new Error(
      `Invalid namespace: ${namespace}. Use dns, url, oid, x500, or a valid UUID string`
    );
  }
}

function generateNameBasedUuid(namespace: string, name: string, version: 3 | 5): string {
  const nsBytes = uuidToBytes(resolveNamespace(namespace));
  const hash = createHash(version === 3 ? "md5" : "sha1")
    .update(nsBytes)
    .update(name, "utf8")
    .digest();
  const bytes = Uint8Array.from(hash.subarray(0, 16));
  bytes[6] = (bytes[6] & 0x0f) | (version << 4);
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  return bytesToUuid(bytes);
}

function generateUuid1(): string {
  if (!_nodeId) {
    _nodeId = Uint8Array.from(randomBytes(6));
    _nodeId[0] |= 0x01;
  }

  const msecs = Date.now();
  let nsecs = _lastNSecs + 1;
  const dt = msecs - _lastMSecs + (nsecs - _lastNSecs) / 10_000;

  if (dt < 0) {
    _clockseq = (_clockseq + 1) & 0x3fff;
  }
  if (dt < 0 || msecs > _lastMSecs) {
    nsecs = 0;
  }
  if (nsecs >= 10_000) {
    throw new Error("Can't create more than 10M UUIDs/sec");
  }

  _lastMSecs = msecs;
  _lastNSecs = nsecs;

  const timestamp = BigInt(msecs) * 10_000n + UUID_EPOCH_OFFSET + BigInt(nsecs);
  const timeLow = Number(timestamp & 0xffffffffn);
  const timeMid = Number((timestamp >> 32n) & 0xffffn);
  const timeHi = Number((timestamp >> 48n) & 0x0fffn);

  const bytes = new Uint8Array(16);
  bytes[0] = (timeLow >>> 24) & 0xff;
  bytes[1] = (timeLow >>> 16) & 0xff;
  bytes[2] = (timeLow >>> 8) & 0xff;
  bytes[3] = timeLow & 0xff;
  bytes[4] = (timeMid >>> 8) & 0xff;
  bytes[5] = timeMid & 0xff;
  bytes[6] = ((timeHi >>> 8) & 0x0f) | 0x10;
  bytes[7] = timeHi & 0xff;
  bytes[8] = ((_clockseq >>> 8) & 0x3f) | 0x80;
  bytes[9] = _clockseq & 0xff;
  bytes.set(_nodeId, 10);

  return bytesToUuid(bytes);
}

function uuidVariant(uuid: string): string {
  const bytes = uuidToBytes(uuid);
  const value = bytes[8];
  if ((value & 0x80) === 0x00) return "reserved for NCS compatibility";
  if ((value & 0xc0) === 0x80) return "specified in RFC 4122";
  if ((value & 0xe0) === 0xc0) return "reserved for Microsoft compatibility";
  return "reserved for future definition";
}

export class GenerateUUID4Node extends BaseNode {
  static readonly nodeType = "lib.uuid.GenerateUUID4";
  static readonly title = "Generate UUID4";
  static readonly description = "Generate a random UUID (version 4).";

  async process(): Promise<Record<string, unknown>> {
    const bytes = Uint8Array.from(randomBytes(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return { output: bytesToUuid(bytes) };
  }
}

export class GenerateUUID1Node extends BaseNode {
  static readonly nodeType = "lib.uuid.GenerateUUID1";
  static readonly title = "Generate UUID1";
  static readonly description = "Generate a time-based UUID (version 1).";

  async process(): Promise<Record<string, unknown>> {
    return { output: generateUuid1() };
  }
}

export class GenerateUUID3Node extends BaseNode {
  static readonly nodeType = "lib.uuid.GenerateUUID3";
  static readonly title = "Generate UUID3";
  static readonly description = "Generate a name-based UUID using MD5 (version 3).";

  defaults() {
    return { namespace: "dns", name: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const namespace = String(inputs.namespace ?? this._props.namespace ?? "dns");
    const name = String(inputs.name ?? this._props.name ?? "");
    return { output: generateNameBasedUuid(namespace, name, 3) };
  }
}

export class GenerateUUID5Node extends BaseNode {
  static readonly nodeType = "lib.uuid.GenerateUUID5";
  static readonly title = "Generate UUID5";
  static readonly description = "Generate a name-based UUID using SHA-1 (version 5).";

  defaults() {
    return { namespace: "dns", name: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const namespace = String(inputs.namespace ?? this._props.namespace ?? "dns");
    const name = String(inputs.name ?? this._props.name ?? "");
    return { output: generateNameBasedUuid(namespace, name, 5) };
  }
}

export class ParseUUIDNode extends BaseNode {
  static readonly nodeType = "lib.uuid.ParseUUID";
  static readonly title = "Parse UUID";
  static readonly description = "Parse and validate a UUID string.";

  defaults() {
    return { uuid_string: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const uuidInput = String(inputs.uuid_string ?? this._props.uuid_string ?? "");
    try {
      const uuid = normalizeUuid(uuidInput);
      const hex = uuid.replaceAll("-", "");
      return {
        output: {
          uuid,
          version: Number.parseInt(uuid[14], 16),
          variant: uuidVariant(uuid),
          hex,
          int: BigInt(`0x${hex}`).toString(10),
          is_valid: true,
        },
      };
    } catch (error) {
      return {
        output: {
          uuid: uuidInput,
          is_valid: false,
          error: error instanceof Error ? error.message : String(error),
        },
      };
    }
  }
}

export class FormatUUIDNode extends BaseNode {
  static readonly nodeType = "lib.uuid.FormatUUID";
  static readonly title = "Format UUID";
  static readonly description = "Format a UUID string in different representations.";

  defaults() {
    return { uuid_string: "", format: "standard" as UUIDFormat };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const uuidInput = String(inputs.uuid_string ?? this._props.uuid_string ?? "");
    const format = String(inputs.format ?? this._props.format ?? "standard") as UUIDFormat;
    const uuid = normalizeUuid(uuidInput);
    const hex = uuid.replaceAll("-", "");

    if (format === "standard") return { output: uuid };
    if (format === "hex") return { output: hex };
    if (format === "urn") return { output: `urn:uuid:${uuid}` };
    if (format === "int") return { output: BigInt(`0x${hex}`).toString(10) };
    if (format === "bytes_hex") return { output: Buffer.from(hex, "hex").toString("hex") };
    throw new Error(`Unsupported format: ${format}`);
  }
}

export class IsValidUUIDNode extends BaseNode {
  static readonly nodeType = "lib.uuid.IsValidUUID";
  static readonly title = "Is Valid UUID";
  static readonly description = "Check if a string is a valid UUID.";

  defaults() {
    return { uuid_string: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const uuidInput = String(inputs.uuid_string ?? this._props.uuid_string ?? "");
    try {
      normalizeUuid(uuidInput);
      return { output: true };
    } catch {
      return { output: false };
    }
  }
}

export const UUID_NODES = [
  GenerateUUID4Node,
  GenerateUUID1Node,
  GenerateUUID3Node,
  GenerateUUID5Node,
  ParseUUIDNode,
  FormatUUIDNode,
  IsValidUUIDNode,
] as const;
