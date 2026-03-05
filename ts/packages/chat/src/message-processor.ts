/**
 * Core chat processing loop with streaming and tool calling.
 *
 * Port of src/nodetool/chat/regular_chat.py (process_regular_chat).
 */

import type { BaseProvider } from "@nodetool/runtime";
import type { Message, ToolCall, ProviderStreamItem } from "@nodetool/runtime";
import type { ProcessingContext } from "@nodetool/runtime";
import type { Chunk } from "@nodetool/protocol";
import type { Tool } from "@nodetool/agents";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatCallbacks {
  /** Called for each text chunk streamed from the provider. */
  onChunk?: (text: string) => void;
  /** Called when a tool call is received from the provider. */
  onToolCall?: (toolCall: ToolCall) => void;
  /** Called after a tool has been executed. */
  onToolResult?: (toolCall: ToolCall, result: unknown) => void;
}

// ---------------------------------------------------------------------------
// Tool runner
// ---------------------------------------------------------------------------

/**
 * Find and execute a tool by name, returning the ToolCall updated with the result.
 */
export async function runTool(
  context: ProcessingContext,
  toolCall: ToolCall,
  tools: Tool[],
): Promise<ToolCall> {
  const tool = tools.find((t) => t.name === toolCall.name);
  if (!tool) {
    throw new Error(`Tool "${toolCall.name}" not found`);
  }

  const result = await tool.process(context, toolCall.args);

  return {
    id: toolCall.id,
    name: toolCall.name,
    args: toolCall.args,
    result,
  } as ToolCall & { result: unknown };
}

// ---------------------------------------------------------------------------
// Chat processing loop
// ---------------------------------------------------------------------------

function isChunk(item: ProviderStreamItem): item is Chunk {
  return "type" in item && (item as Chunk).type === "chunk";
}

function isToolCall(item: ProviderStreamItem): item is ToolCall {
  return "name" in item && "id" in item && !("type" in item);
}

/**
 * Serializer that handles objects with a `toJSON` method or falls back to
 * stringification, similar to the Python `default_serializer`.
 */
function defaultSerializer(_key: string, value: unknown): unknown {
  if (value !== null && typeof value === "object" && "toJSON" in value) {
    return (value as { toJSON: () => unknown }).toJSON();
  }
  return value;
}

/**
 * Process a user message through the provider with streaming and tool calling.
 *
 * Implements the core loop from `process_regular_chat`:
 * 1. Append user message.
 * 2. Stream provider response, accumulating text chunks into an assistant message.
 * 3. When tool calls are received, execute each tool and append assistant + tool messages.
 * 4. If tool calls were processed, re-send the new messages to get the next response.
 * 5. When no more tool calls are pending, return the full message history.
 */
export async function processChat(opts: {
  userInput: string;
  messages: Message[];
  model: string;
  provider: BaseProvider;
  context: ProcessingContext;
  tools?: Tool[];
  callbacks?: ChatCallbacks;
}): Promise<Message[]> {
  const { userInput, messages, model, provider, context, tools = [], callbacks } = opts;

  // 1. Add user message
  messages.push({ role: "user", content: userInput });

  const providerTools =
    tools.length > 0 ? tools.map((t) => t.toProviderTool()) : undefined;

  let messagesToSend: Message[] = messages;

  while (true) {
    let unprocessedMessages: Message[] = [];

    const stream = provider.generateMessages({
      messages: messagesToSend,
      model,
      tools: providerTools,
    });

    for await (const item of stream) {
      // --- Text chunk ---
      if (isChunk(item)) {
        const text = item.content ?? "";
        callbacks?.onChunk?.(text);

        const last = messages[messages.length - 1];
        if (last && last.role === "assistant" && typeof last.content === "string") {
          last.content += text;
        } else {
          messages.push({ role: "assistant", content: text });
        }
      }

      // --- Tool call ---
      if (isToolCall(item)) {
        callbacks?.onToolCall?.(item);

        const toolResult = await runTool(context, item, tools);
        callbacks?.onToolResult?.(item, (toolResult as ToolCall & { result: unknown }).result);

        // Append assistant message with tool call
        unprocessedMessages.push({
          role: "assistant",
          toolCalls: [item],
        });

        // Append tool result message
        unprocessedMessages.push({
          role: "tool",
          toolCallId: item.id,
          content: JSON.stringify(
            (toolResult as ToolCall & { result: unknown }).result,
            defaultSerializer,
          ),
        });
      }
    }

    // If tool calls were processed, continue the conversation
    if (unprocessedMessages.length > 0) {
      messages.push(...unprocessedMessages);
      messagesToSend = unprocessedMessages;
    } else {
      break;
    }
  }

  return messages;
}
