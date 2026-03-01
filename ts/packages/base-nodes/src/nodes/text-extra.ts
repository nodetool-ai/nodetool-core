import { BaseNode } from "@nodetool/node-sdk";

function flagsFromOpts(opts: {
  dotall?: unknown;
  ignorecase?: unknown;
  multiline?: unknown;
}): string {
  let flags = "";
  if (opts.ignorecase) flags += "i";
  if (opts.multiline) flags += "m";
  if (opts.dotall) flags += "s";
  return flags;
}

function toTitleCase(value: string): string {
  return value.replace(/\w\S*/g, (txt) =>
    txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase()
  );
}

export class SplitTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Split";
  static readonly title = "Split Text";
  static readonly description = "Split text by delimiter";

  defaults() {
    return { text: "", delimiter: "," };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const delimiter = String(inputs.delimiter ?? this._props.delimiter ?? ",");
    return { output: text.split(delimiter) };
  }
}

export class ExtractTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Extract";
  static readonly title = "Extract Text";
  static readonly description = "Extract substring by start/end";

  defaults() {
    return { text: "", start: 0, end: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const end = Number(inputs.end ?? this._props.end ?? 0);
    return { output: text.slice(start, end) };
  }
}

export class ChunkTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Chunk";
  static readonly title = "Split Text into Chunks";
  static readonly description = "Split text into chunked windows";

  defaults() {
    return { text: "", length: 100, overlap: 0, separator: " " };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const length = Number(inputs.length ?? this._props.length ?? 100);
    const overlap = Number(inputs.overlap ?? this._props.overlap ?? 0);
    const separator = String(inputs.separator ?? this._props.separator ?? " ");

    const step = length - overlap;
    if (length < 1 || step <= 0) {
      throw new Error("Invalid chunk parameters");
    }

    const words = text.split(separator);
    const chunks: string[] = [];
    for (let i = 0; i < words.length; i += step) {
      chunks.push(words.slice(i, i + length).join(" "));
    }
    return { output: chunks };
  }
}

export class ExtractRegexNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ExtractRegex";
  static readonly title = "Extract Regex Groups";
  static readonly description = "Extract capture groups from first regex match";

  defaults() {
    return {
      text: "",
      regex: "",
      dotall: false,
      ignorecase: false,
      multiline: false,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.regex ?? this._props.regex ?? "");
    const flags = flagsFromOpts({
      dotall: inputs.dotall ?? this._props.dotall,
      ignorecase: inputs.ignorecase ?? this._props.ignorecase,
      multiline: inputs.multiline ?? this._props.multiline,
    });

    const match = new RegExp(pattern, flags).exec(text);
    if (!match) {
      return { output: [] };
    }
    return { output: match.slice(1) };
  }
}

export class FindAllRegexNode extends BaseNode {
  static readonly nodeType = "nodetool.text.FindAllRegex";
  static readonly title = "Find All Regex Matches";
  static readonly description = "Find all regex matches in text";

  defaults() {
    return {
      text: "",
      regex: "",
      dotall: false,
      ignorecase: false,
      multiline: false,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.regex ?? this._props.regex ?? "");
    const flags = `${flagsFromOpts({
      dotall: inputs.dotall ?? this._props.dotall,
      ignorecase: inputs.ignorecase ?? this._props.ignorecase,
      multiline: inputs.multiline ?? this._props.multiline,
    })}g`;

    const matches = [...text.matchAll(new RegExp(pattern, flags))].map((m) => m[0]);
    return { output: matches };
  }
}

export class TextParseJSONNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ParseJSON";
  static readonly title = "Parse JSON String";
  static readonly description = "Parse JSON string to object";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    return { output: JSON.parse(text) };
  }
}

export class RegexMatchNode extends BaseNode {
  static readonly nodeType = "nodetool.text.RegexMatch";
  static readonly title = "Find Regex Matches";
  static readonly description = "Find regex matches and return selected group";

  defaults() {
    return { text: "", pattern: "", group: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "");
    const group = inputs.group ?? this._props.group;

    if (group === null || group === undefined) {
      return { output: [...text.matchAll(new RegExp(pattern, "g"))].map((m) => m[0]) };
    }

    const groupIndex = Number(group);
    const out = [...text.matchAll(new RegExp(pattern, "g"))].map((m) => m[groupIndex]);
    return { output: out.filter((v) => v !== undefined) };
  }
}

export class RegexReplaceNode extends BaseNode {
  static readonly nodeType = "nodetool.text.RegexReplace";
  static readonly title = "Replace with Regex";
  static readonly description = "Regex text replacement";

  defaults() {
    return { text: "", pattern: "", replacement: "", count: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    let text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "");
    const replacement = String(inputs.replacement ?? this._props.replacement ?? "");
    const count = Number(inputs.count ?? this._props.count ?? 0);

    if (count <= 0) {
      return { output: text.replace(new RegExp(pattern, "g"), replacement) };
    }

    const regex = new RegExp(pattern, "g");
    let replaced = 0;
    text = text.replace(regex, (match) => {
      if (replaced >= count) {
        return match;
      }
      replaced += 1;
      return replacement;
    });
    return { output: text };
  }
}

export class RegexSplitNode extends BaseNode {
  static readonly nodeType = "nodetool.text.RegexSplit";
  static readonly title = "Split with Regex";
  static readonly description = "Split text by regex pattern";

  defaults() {
    return { text: "", pattern: "", maxsplit: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "");
    const maxsplit = Number(inputs.maxsplit ?? this._props.maxsplit ?? 0);

    const split = text.split(new RegExp(pattern));
    if (maxsplit <= 0) {
      return { output: split };
    }
    return { output: [split.slice(0, maxsplit).join(""), ...split.slice(maxsplit)] };
  }
}

export class RegexValidateNode extends BaseNode {
  static readonly nodeType = "nodetool.text.RegexValidate";
  static readonly title = "Validate with Regex";
  static readonly description = "Validate text against regex pattern";

  defaults() {
    return { text: "", pattern: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "");
    return { output: new RegExp(pattern).test(text) };
  }
}

export class CompareTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Compare";
  static readonly title = "Compare Text";
  static readonly description = "Compare two texts lexically";

  defaults() {
    return { text_a: "", text_b: "", case_sensitive: true, trim_whitespace: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const textA = String(inputs.text_a ?? this._props.text_a ?? "");
    const textB = String(inputs.text_b ?? this._props.text_b ?? "");
    const caseSensitive = Boolean(
      inputs.case_sensitive ?? this._props.case_sensitive ?? true
    );
    const trimWhitespace = Boolean(
      inputs.trim_whitespace ?? this._props.trim_whitespace ?? false
    );

    const normalize = (value: string) => {
      const trimmed = trimWhitespace ? value.trim() : value;
      return caseSensitive ? trimmed : trimmed.toLowerCase();
    };

    const left = normalize(textA);
    const right = normalize(textB);

    if (left < right) {
      return { output: "less" };
    }
    if (left > right) {
      return { output: "greater" };
    }
    return { output: "equal" };
  }
}

export class EqualsTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Equals";
  static readonly title = "Equals";
  static readonly description = "Check text equality";

  defaults() {
    return { text_a: "", text_b: "", case_sensitive: true, trim_whitespace: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const cmp = new CompareTextNode();
    cmp.assign(this.defaults());
    cmp.assign(this._props);
    const result = await cmp.process(inputs);
    return { output: result.output === "equal" };
  }
}

export class ToUppercaseNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ToUppercase";
  static readonly title = "To Uppercase";
  static readonly description = "Convert text to uppercase";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: String(inputs.text ?? this._props.text ?? "").toUpperCase() };
  }
}

export class ToLowercaseNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ToLowercase";
  static readonly title = "To Lowercase";
  static readonly description = "Convert text to lowercase";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: String(inputs.text ?? this._props.text ?? "").toLowerCase() };
  }
}

export class ToTitlecaseNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ToTitlecase";
  static readonly title = "To Title Case";
  static readonly description = "Convert text to title case";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: toTitleCase(String(inputs.text ?? this._props.text ?? "")) };
  }
}

export class CapitalizeTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.CapitalizeText";
  static readonly title = "Capitalize Text";
  static readonly description = "Capitalize first character";

  defaults() {
    return { text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    if (!text) {
      return { output: "" };
    }
    return { output: text.charAt(0).toUpperCase() + text.slice(1).toLowerCase() };
  }
}

export class SliceTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Slice";
  static readonly title = "Slice Text";
  static readonly description = "Slice text by start/stop/step";

  defaults() {
    return { text: "", start: 0, stop: 0, step: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const stop = Number(inputs.stop ?? this._props.stop ?? 0);
    const step = Number(inputs.step ?? this._props.step ?? 1);

    if (step === 0) {
      throw new Error("slice step cannot be zero");
    }

    if (step === 1) {
      return { output: text.slice(start, stop) };
    }

    const chars = [...text];
    const result: string[] = [];
    const len = chars.length;
    const normStart = start < 0 ? len + start : start;
    const normStop = stop < 0 ? len + stop : stop;

    if (step > 0) {
      for (let i = Math.max(0, normStart); i < Math.min(len, normStop); i += step) {
        result.push(chars[i]);
      }
    } else {
      for (let i = Math.min(len - 1, normStart); i > Math.max(-1, normStop); i += step) {
        if (i >= 0 && i < len) {
          result.push(chars[i]);
        }
      }
    }

    return { output: result.join("") };
  }
}

export class StartsWithTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.StartsWith";
  static readonly title = "Starts With";
  static readonly description = "Check if text starts with prefix";

  defaults() {
    return { text: "", prefix: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const prefix = String(inputs.prefix ?? this._props.prefix ?? "");
    return { output: text.startsWith(prefix) };
  }
}

export class EndsWithTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.EndsWith";
  static readonly title = "Ends With";
  static readonly description = "Check if text ends with suffix";

  defaults() {
    return { text: "", suffix: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const suffix = String(inputs.suffix ?? this._props.suffix ?? "");
    return { output: text.endsWith(suffix) };
  }
}

export class ContainsTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Contains";
  static readonly title = "Contains Text";
  static readonly description = "Check substring membership with any/all/none";

  defaults() {
    return {
      text: "",
      substring: "",
      search_values: [] as string[],
      case_sensitive: true,
      match_mode: "any",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const substring = String(inputs.substring ?? this._props.substring ?? "");
    const searchValues = Array.isArray(inputs.search_values ?? this._props.search_values)
      ? ((inputs.search_values ?? this._props.search_values) as unknown[]).map((v) =>
          String(v)
        )
      : [];
    const caseSensitive = Boolean(
      inputs.case_sensitive ?? this._props.case_sensitive ?? true
    );
    const matchMode = String(inputs.match_mode ?? this._props.match_mode ?? "any");

    const targets = searchValues.length > 0 ? searchValues : substring ? [substring] : [];
    if (targets.length === 0) {
      return { output: false };
    }

    const haystack = caseSensitive ? text : text.toLowerCase();
    const needles = caseSensitive ? targets : targets.map((n) => n.toLowerCase());

    if (matchMode === "all") {
      return { output: needles.every((needle) => haystack.includes(needle)) };
    }
    if (matchMode === "none") {
      return { output: needles.every((needle) => !haystack.includes(needle)) };
    }
    return { output: needles.some((needle) => haystack.includes(needle)) };
  }
}

export class TrimWhitespaceNode extends BaseNode {
  static readonly nodeType = "nodetool.text.TrimWhitespace";
  static readonly title = "Trim Whitespace";
  static readonly description = "Trim whitespace from text edges";

  defaults() {
    return { text: "", trim_start: true, trim_end: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const trimStart = Boolean(inputs.trim_start ?? this._props.trim_start ?? true);
    const trimEnd = Boolean(inputs.trim_end ?? this._props.trim_end ?? true);

    if (trimStart && trimEnd) {
      return { output: text.trim() };
    }
    if (trimStart) {
      return { output: text.replace(/^\s+/, "") };
    }
    if (trimEnd) {
      return { output: text.replace(/\s+$/, "") };
    }
    return { output: text };
  }
}

export class CollapseWhitespaceNode extends BaseNode {
  static readonly nodeType = "nodetool.text.CollapseWhitespace";
  static readonly title = "Collapse Whitespace";
  static readonly description = "Collapse whitespace runs";

  defaults() {
    return {
      text: "",
      preserve_newlines: false,
      replacement: " ",
      trim_edges: true,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const preserveNewlines = Boolean(
      inputs.preserve_newlines ?? this._props.preserve_newlines ?? false
    );
    const replacement = String(inputs.replacement ?? this._props.replacement ?? " ");
    const trimEdges = Boolean(inputs.trim_edges ?? this._props.trim_edges ?? true);

    const value = trimEdges ? text.trim() : text;
    if (preserveNewlines) {
      return { output: value.replace(/[^\S\r\n]+/g, replacement) };
    }
    return { output: value.replace(/\s+/g, replacement) };
  }
}

export class IsEmptyTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.IsEmpty";
  static readonly title = "Is Empty";
  static readonly description = "Check whether text is empty";

  defaults() {
    return { text: "", trim_whitespace: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const trimWhitespace = Boolean(
      inputs.trim_whitespace ?? this._props.trim_whitespace ?? true
    );
    return { output: (trimWhitespace ? text.trim() : text).length === 0 };
  }
}

export class RemovePunctuationNode extends BaseNode {
  static readonly nodeType = "nodetool.text.RemovePunctuation";
  static readonly title = "Remove Punctuation";
  static readonly description = "Remove configured punctuation characters";

  defaults() {
    return {
      text: "",
      replacement: "",
      punctuation: `!"#$%&'()*+,\\-./:;<=>?@[\\]^_{|}~`,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const replacement = String(inputs.replacement ?? this._props.replacement ?? "");
    const punctuation = String(
      inputs.punctuation ?? this._props.punctuation ?? `!"#$%&'()*+,\\-./:;<=>?@[\\]^_{|}~`
    );

    const escaped = punctuation.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return { output: text.replace(new RegExp(`[${escaped}]`, "g"), replacement) };
  }
}

export class StripAccentsNode extends BaseNode {
  static readonly nodeType = "nodetool.text.StripAccents";
  static readonly title = "Strip Accents";
  static readonly description = "Remove accent marks from text";

  defaults() {
    return { text: "", preserve_non_ascii: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const preserveNonAscii = Boolean(
      inputs.preserve_non_ascii ?? this._props.preserve_non_ascii ?? true
    );

    const normalized = text.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    if (preserveNonAscii) {
      return { output: normalized };
    }
    return { output: normalized.replace(/[^\x00-\x7F]/g, "") };
  }
}

export class SlugifyNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Slugify";
  static readonly title = "Slugify";
  static readonly description = "Convert text to URL-safe slug";

  defaults() {
    return { text: "", separator: "-", lowercase: true, allow_unicode: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const separator = String(inputs.separator ?? this._props.separator ?? "-");
    const lowercase = Boolean(inputs.lowercase ?? this._props.lowercase ?? true);
    const allowUnicode = Boolean(
      inputs.allow_unicode ?? this._props.allow_unicode ?? false
    );

    let value = text.normalize("NFKD");
    if (!allowUnicode) {
      value = value.replace(/[^\x00-\x7F]/g, "");
    }
    value = value.replace(/[^\w\s-]/g, "");
    value = value.replace(/[\s_-]+/g, separator).replace(/^[-_]+|[-_]+$/g, "");
    if (lowercase) {
      value = value.toLowerCase();
    }
    return { output: value };
  }
}

export class HasLengthTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.HasLength";
  static readonly title = "Check Length";
  static readonly description = "Check text length constraints";

  defaults() {
    return { text: "", min_length: 0, max_length: 0, exact_length: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const minLength = Number(inputs.min_length ?? this._props.min_length ?? 0);
    const maxLength = Number(inputs.max_length ?? this._props.max_length ?? 0);
    const exactLength = Number(inputs.exact_length ?? this._props.exact_length ?? 0);

    const length = text.length;

    if (exactLength !== null) {
      return { output: length === exactLength };
    }
    if (minLength !== null && length < minLength) {
      return { output: false };
    }
    if (maxLength !== null && length > maxLength) {
      return { output: false };
    }
    return { output: true };
  }
}

export class TruncateTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.TruncateText";
  static readonly title = "Truncate Text";
  static readonly description = "Truncate text to max length";

  defaults() {
    return { text: "", max_length: 100, ellipsis: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const maxLength = Number(inputs.max_length ?? this._props.max_length ?? 100);
    const ellipsis = String(inputs.ellipsis ?? this._props.ellipsis ?? "");

    if (maxLength <= 0) {
      return { output: ellipsis || "" };
    }
    if (text.length <= maxLength) {
      return { output: text };
    }
    if (ellipsis && ellipsis.length < maxLength) {
      const cut = maxLength - ellipsis.length;
      return { output: `${text.slice(0, cut)}${ellipsis}` };
    }
    return { output: text.slice(0, maxLength) };
  }
}

export class PadTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.PadText";
  static readonly title = "Pad Text";
  static readonly description = "Pad text to target length";

  defaults() {
    return { text: "", length: 0, pad_character: " ", direction: "right" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const length = Number(inputs.length ?? this._props.length ?? 0);
    const padCharacter = String(inputs.pad_character ?? this._props.pad_character ?? " ");
    const direction = String(inputs.direction ?? this._props.direction ?? "right");

    if (padCharacter.length !== 1) {
      throw new Error("pad_character must be a single character");
    }
    if (length <= text.length) {
      return { output: text };
    }

    const needed = length - text.length;
    if (direction === "left") {
      return { output: padCharacter.repeat(needed) + text };
    }
    if (direction === "both") {
      const left = Math.floor(needed / 2);
      const right = needed - left;
      return { output: `${padCharacter.repeat(left)}${text}${padCharacter.repeat(right)}` };
    }
    return { output: text + padCharacter.repeat(needed) };
  }
}

export class LengthTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Length";
  static readonly title = "Measure Length";
  static readonly description = "Measure text as chars/words/lines";

  defaults() {
    return { text: "", measure: "characters", trim_whitespace: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const measure = String(inputs.measure ?? this._props.measure ?? "characters");
    const trimWhitespace = Boolean(
      inputs.trim_whitespace ?? this._props.trim_whitespace ?? false
    );

    const value = trimWhitespace ? text.trim() : text;

    if (measure === "words") {
      return { output: value.split(/\s+/).filter(Boolean).length };
    }
    if (measure === "lines") {
      if (!value) {
        return { output: 0 };
      }
      return {
        output: value
          .split(/\r?\n/)
          .filter((line) => line || !trimWhitespace).length,
      };
    }
    return { output: value.length };
  }
}

export class IndexOfTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.IndexOf";
  static readonly title = "Index Of";
  static readonly description = "Find substring index in text";

  defaults() {
    return {
      text: "",
      substring: "",
      case_sensitive: true,
      start_index: 0,
      end_index: 0,
      search_from_end: false,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    let haystack = String(inputs.text ?? this._props.text ?? "");
    let needle = String(inputs.substring ?? this._props.substring ?? "");
    const caseSensitive = Boolean(
      inputs.case_sensitive ?? this._props.case_sensitive ?? true
    );
    const startIndex = Number(inputs.start_index ?? this._props.start_index ?? 0);
    const endIndex = Number(inputs.end_index ?? this._props.end_index ?? 0);
    const searchFromEnd = Boolean(
      inputs.search_from_end ?? this._props.search_from_end ?? false
    );

    if (!caseSensitive) {
      haystack = haystack.toLowerCase();
      needle = needle.toLowerCase();
    }

    const end = Math.max(startIndex, endIndex);

    if (searchFromEnd) {
      return { output: haystack.lastIndexOf(needle, end) };
    }

    const idx = haystack.slice(startIndex, end).indexOf(needle);
    return { output: idx < 0 ? -1 : startIndex + idx };
  }
}

export class SurroundWithTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.SurroundWith";
  static readonly title = "Surround With";
  static readonly description = "Wrap text with prefix and suffix";

  defaults() {
    return { text: "", prefix: "", suffix: "", skip_if_wrapped: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const prefix = String(inputs.prefix ?? this._props.prefix ?? "");
    const suffix = String(inputs.suffix ?? this._props.suffix ?? "");
    const skipIfWrapped = Boolean(
      inputs.skip_if_wrapped ?? this._props.skip_if_wrapped ?? true
    );

    if (skipIfWrapped && text.startsWith(prefix) && text.endsWith(suffix)) {
      return { output: text };
    }
    return { output: `${prefix}${text}${suffix}` };
  }
}

type FilterStringType =
  | "contains"
  | "starts_with"
  | "ends_with"
  | "length_greater"
  | "length_less"
  | "exact_length";

export class FilterStringNode extends BaseNode {
  static readonly nodeType = "nodetool.text.FilterString";
  static readonly title = "Filter String";
  static readonly description = "Stream-filter strings by criteria";
  static readonly syncMode = "on_any" as const;

  private _filterType: FilterStringType = "contains";
  private _criteria = "";

  defaults() {
    return { value: "", filter_type: "contains", criteria: "" };
  }

  async initialize(): Promise<void> {
    this._filterType = String(this._props.filter_type ?? "contains") as FilterStringType;
    this._criteria = String(this._props.criteria ?? "");
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("filter_type" in inputs) {
      this._filterType = String(inputs.filter_type ?? "contains") as FilterStringType;
      return {};
    }
    if ("criteria" in inputs) {
      this._criteria = String(inputs.criteria ?? "");
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const value = inputs.value;
    if (typeof value !== "string") {
      return {};
    }

    const criteria = this._criteria;
    const len = value.length;
    const n = Number(criteria);

    let matched = false;
    switch (this._filterType) {
      case "contains":
        matched = value.includes(criteria);
        break;
      case "starts_with":
        matched = value.startsWith(criteria);
        break;
      case "ends_with":
        matched = value.endsWith(criteria);
        break;
      case "length_greater":
        matched = Number.isFinite(n) && len > n;
        break;
      case "length_less":
        matched = Number.isFinite(n) && len < n;
        break;
      case "exact_length":
        matched = Number.isFinite(n) && len === n;
        break;
      default:
        matched = false;
    }

    if (!matched) {
      return {};
    }
    return { output: value };
  }
}

export class FilterRegexStringNode extends BaseNode {
  static readonly nodeType = "nodetool.text.FilterRegexString";
  static readonly title = "Filter Regex String";
  static readonly description = "Stream-filter strings by regex";
  static readonly syncMode = "on_any" as const;

  private _pattern = "";
  private _fullMatch = false;

  defaults() {
    return { value: "", pattern: "", full_match: false };
  }

  async initialize(): Promise<void> {
    this._pattern = String(this._props.pattern ?? "");
    this._fullMatch = Boolean(this._props.full_match ?? false);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("pattern" in inputs) {
      this._pattern = String(inputs.pattern ?? "");
      return {};
    }
    if ("full_match" in inputs) {
      this._fullMatch = Boolean(inputs.full_match);
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const value = inputs.value;
    if (typeof value !== "string") {
      return {};
    }

    let regex: RegExp;
    try {
      regex = new RegExp(this._pattern);
    } catch {
      return {};
    }

    const matched = this._fullMatch
      ? (value.match(regex)?.[0] ?? "") === value
      : regex.test(value);

    if (!matched) {
      return {};
    }
    return { output: value };
  }
}

export const TEXT_EXTRA_NODES = [
  SplitTextNode,
  ExtractTextNode,
  ChunkTextNode,
  ExtractRegexNode,
  FindAllRegexNode,
  TextParseJSONNode,
  RegexMatchNode,
  RegexReplaceNode,
  RegexSplitNode,
  RegexValidateNode,
  CompareTextNode,
  EqualsTextNode,
  ToUppercaseNode,
  ToLowercaseNode,
  ToTitlecaseNode,
  CapitalizeTextNode,
  SliceTextNode,
  StartsWithTextNode,
  EndsWithTextNode,
  ContainsTextNode,
  TrimWhitespaceNode,
  CollapseWhitespaceNode,
  IsEmptyTextNode,
  RemovePunctuationNode,
  StripAccentsNode,
  SlugifyNode,
  HasLengthTextNode,
  TruncateTextNode,
  PadTextNode,
  LengthTextNode,
  IndexOfTextNode,
  SurroundWithTextNode,
  FilterStringNode,
  FilterRegexStringNode,
] as const;
