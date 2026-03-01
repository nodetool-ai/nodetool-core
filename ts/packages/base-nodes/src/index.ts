import type { NodeRegistry } from "@nodetool/node-sdk";

export {
  IfNode,
  ForEachNode,
  CollectNode,
  RerouteNode,
  CONTROL_NODES,
} from "./nodes/control.js";
export {
  ConditionalSwitchNode,
  LogicalOperatorNode,
  NotNode,
  CompareNode,
  IsNoneNode,
  IsInNode,
  AllNode,
  SomeNode,
  BOOLEAN_NODES,
} from "./nodes/boolean.js";
export {
  LengthNode,
  ListRangeNode,
  GenerateSequenceNode,
  SliceNode,
  SelectElementsNode,
  GetElementNode,
  AppendNode,
  ExtendNode,
  DedupeNode,
  ReverseNode,
  RandomizeNode,
  SortNode,
  IntersectionNode,
  UnionNode,
  DifferenceNode,
  ChunkNode,
  SumNode,
  AverageNode,
  MinimumNode,
  MaximumNode,
  ProductNode,
  FlattenNode,
  LIST_NODES,
} from "./nodes/list.js";
export {
  ToStringNode,
  ConcatTextNode,
  JoinTextNode,
  ReplaceTextNode,
  CollectTextNode,
  TEXT_NODES,
} from "./nodes/text.js";
export {
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
  TEXT_EXTRA_NODES,
} from "./nodes/text-extra.js";
export {
  ConstantBoolNode,
  ConstantIntegerNode,
  ConstantFloatNode,
  ConstantStringNode,
  ConstantListNode,
  ConstantTextListNode,
  ConstantDictNode,
  CONSTANT_NODES,
} from "./nodes/constant.js";
export {
  FilterNumberNode,
  FilterNumberRangeNode,
  NUMBERS_NODES,
} from "./nodes/numbers.js";
export {
  GetValueNode,
  UpdateDictionaryNode,
  RemoveDictionaryKeyNode,
  ParseJSONNode,
  ZipDictionaryNode,
  CombineDictionaryNode,
  FilterDictionaryNode,
  ReduceDictionariesNode,
  MakeDictionaryNode,
  ArgMaxNode,
  ToJSONNode,
  FilterDictByNumberNode,
  FilterDictByRangeNode,
  FilterDictRegexNode,
  FilterDictByValueNode,
  DICTIONARY_NODES,
} from "./nodes/dictionary.js";

import { CONTROL_NODES } from "./nodes/control.js";
import { BOOLEAN_NODES } from "./nodes/boolean.js";
import { LIST_NODES } from "./nodes/list.js";
import { TEXT_NODES } from "./nodes/text.js";
import { TEXT_EXTRA_NODES } from "./nodes/text-extra.js";
import { CONSTANT_NODES } from "./nodes/constant.js";
import { NUMBERS_NODES } from "./nodes/numbers.js";
import { DICTIONARY_NODES } from "./nodes/dictionary.js";

export const ALL_BASE_NODES = [
  ...CONTROL_NODES,
  ...BOOLEAN_NODES,
  ...LIST_NODES,
  ...TEXT_NODES,
  ...TEXT_EXTRA_NODES,
  ...CONSTANT_NODES,
  ...NUMBERS_NODES,
  ...DICTIONARY_NODES,
] as const;

export function registerBaseNodes(registry: NodeRegistry): void {
  for (const nodeClass of ALL_BASE_NODES) {
    registry.register(nodeClass);
  }
}
