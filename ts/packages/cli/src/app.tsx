/**
 * NodeTool Chat CLI — Ink-based terminal UI.
 *
 * Layout:
 *   ┌── nodetool chat ───────────────────────────────────────┐
 *   │ provider: anthropic • model: claude-... • agent: OFF   │
 *   ├────────────────────────────────────────────────────────┤
 *   │ [Static: past messages rendered with markdown]         │
 *   │                                                        │
 *   │ [Streaming: live token output + spinner]               │
 *   ├────────────────────────────────────────────────────────┤
 *   │ > user input with history                              │
 *   │   ↑↓ history  tab complete  /help for commands         │
 *   └────────────────────────────────────────────────────────┘
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Box, Text, Static, useApp, useInput } from "ink";
import TextInput from "ink-text-input";
import Spinner from "ink-spinner";
import type { Message, ToolCall } from "@nodetool/runtime";
import { ProcessingContext } from "@nodetool/runtime";
import { processChat } from "@nodetool/chat";
import { Agent } from "@nodetool/agents";
import { ReadFileTool, WriteFileTool, ListDirectoryTool } from "@nodetool/agents";
import { createProvider, DEFAULT_MODELS } from "./providers.js";
import { renderMarkdown } from "./markdown.js";
import { saveSettings } from "./settings.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  toolName?: string;
  rendered?: string; // pre-rendered markdown
}

interface AppProps {
  initialProvider: string;
  initialModel: string;
  initialAgentMode: boolean;
  enabledTools: string[];
  workspaceDir: string;
}

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

const COMMANDS = {
  "/help":     "Show available commands",
  "/clear":    "Clear conversation history",
  "/model":    "Set model: /model <model-id>",
  "/provider": "Set provider: /provider <name>",
  "/agent":    "Toggle agent mode",
  "/tools":    "List enabled tools",
  "/exit":     "Exit the chat",
  "/quit":     "Exit the chat",
} as const;

const COMMAND_NAMES = Object.keys(COMMANDS);

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function StatusBar({
  provider,
  model,
  agentMode,
  msgCount,
  streaming,
}: {
  provider: string;
  model: string;
  agentMode: boolean;
  msgCount: number;
  streaming: boolean;
}) {
  const width = process.stdout.columns ?? 80;
  const left = `  nodetool chat  `;
  const mid = `${provider} • ${model}`;
  const right = `  agent: ${agentMode ? "ON " : "OFF"}  msgs: ${msgCount}  `;
  const pad = Math.max(0, width - left.length - mid.length - right.length);

  return (
    <Box>
      <Text backgroundColor="blue" color="white" bold>{left}</Text>
      <Text backgroundColor="black" color="cyan">{mid}</Text>
      <Text backgroundColor="black" color="gray">{" ".repeat(pad)}</Text>
      {streaming
        ? <Text backgroundColor="black" color="yellow"><Spinner type="dots" /> streaming</Text>
        : <Text backgroundColor="blue" color="white">{right}</Text>}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Individual message rendering
// ---------------------------------------------------------------------------

function UserMessage({ content }: { content: string }) {
  const width = process.stdout.columns ?? 80;
  const prefix = "  You  ";
  const padded = "─".repeat(Math.max(0, width - prefix.length - 2));
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color="cyan" bold>{prefix}</Text>
        <Text color="gray" dimColor>{padded}</Text>
      </Box>
      <Box marginLeft={2} marginBottom={1}>
        <Text>{content}</Text>
      </Box>
    </Box>
  );
}

function AssistantMessage({ content, rendered }: { content: string; rendered?: string }) {
  const width = process.stdout.columns ?? 80;
  const prefix = "  Assistant  ";
  const padded = "─".repeat(Math.max(0, width - prefix.length - 2));
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color="green" bold>{prefix}</Text>
        <Text color="gray" dimColor>{padded}</Text>
      </Box>
      <Box marginLeft={2} marginBottom={1}>
        <Text>{rendered ?? content}</Text>
      </Box>
    </Box>
  );
}

function ToolMessage({ toolName, content }: { toolName: string; content: string }) {
  return (
    <Box marginLeft={2} marginBottom={1}>
      <Text color="yellow">⚙ {toolName}: </Text>
      <Text color="gray" dimColor>{content.slice(0, 120)}{content.length > 120 ? "…" : ""}</Text>
    </Box>
  );
}

function SystemMessage({ content }: { content: string }) {
  return (
    <Box marginLeft={2} marginBottom={1}>
      <Text color="magenta" italic>{content}</Text>
    </Box>
  );
}

function ChatMessageItem({ msg }: { msg: ChatMessage }) {
  switch (msg.role) {
    case "user":      return <UserMessage content={msg.content} />;
    case "assistant": return <AssistantMessage content={msg.content} rendered={msg.rendered} />;
    case "tool":      return <ToolMessage toolName={msg.toolName ?? "tool"} content={msg.content} />;
    case "system":    return <SystemMessage content={msg.content} />;
  }
}

// ---------------------------------------------------------------------------
// Help panel
// ---------------------------------------------------------------------------

function HelpPanel() {
  return (
    <Box flexDirection="column" marginLeft={2} marginTop={1} marginBottom={1}>
      <Text bold color="cyan">Available commands:</Text>
      {Object.entries(COMMANDS).map(([cmd, desc]) => (
        <Box key={cmd}>
          <Text color="yellow">{cmd.padEnd(14)}</Text>
          <Text color="gray">{desc}</Text>
        </Box>
      ))}
      <Box marginTop={1}><Text color="gray" dimColor>↑/↓ cycle history • Tab: complete command • Ctrl+C: exit</Text></Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export function App({
  initialProvider,
  initialModel,
  initialAgentMode,
  enabledTools,
  workspaceDir,
}: AppProps) {
  const { exit } = useApp();

  // --- State ---
  const [provider, setProvider] = useState(initialProvider);
  const [model, setModel] = useState(initialModel);
  const [agentMode, setAgentMode] = useState(initialAgentMode);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      content: `Welcome to nodetool chat. Provider: ${initialProvider} • Model: ${initialModel}. Type /help for commands.`,
    },
  ]);
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [inputHistory, setInputHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [historyDraft, setHistoryDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState("");
  const [streamLabel, setStreamLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // Refs to hold latest values without causing re-renders in async callbacks
  const chatHistoryRef = useRef(chatHistory);
  const providerRef = useRef(provider);
  const modelRef = useRef(model);
  const agentModeRef = useRef(agentMode);

  useEffect(() => { chatHistoryRef.current = chatHistory; }, [chatHistory]);
  useEffect(() => { providerRef.current = provider; }, [provider]);
  useEffect(() => { modelRef.current = model; }, [model]);
  useEffect(() => { agentModeRef.current = agentMode; }, [agentMode]);

  // Message counter (excludes welcome/system)
  const msgCount = messages.filter(m => m.role !== "system").length;

  // Unique ID generator
  const nextId = useRef(0);
  const genId = () => `msg-${++nextId.current}`;

  // Add a message to the display history
  const addMessage = useCallback(async (role: ChatMessage["role"], content: string, opts?: { toolName?: string }) => {
    let rendered: string | undefined;
    if (role === "assistant") {
      rendered = await renderMarkdown(content);
    }
    setMessages(prev => [...prev, {
      id: genId(),
      role,
      content,
      rendered,
      toolName: opts?.toolName,
    }]);
  }, []);

  // Create tools from enabled list
  function buildTools() {
    const ctx = new ProcessingContext({ jobId: crypto.randomUUID(), workspaceDir });
    const toolMap: Record<string, unknown> = {
      read_file: new ReadFileTool(),
      write_file: new WriteFileTool(),
      list_directory: new ListDirectoryTool(),
    };
    return enabledTools
      .filter(name => name in toolMap)
      .map(name => toolMap[name] as import("@nodetool/agents").Tool);
  }

  // ---------------------------------------------------------------------------
  // Command handler
  // ---------------------------------------------------------------------------
  const handleCommand = useCallback(async (raw: string): Promise<boolean> => {
    const parts = raw.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    switch (cmd) {
      case "/help":
        setShowHelp(prev => !prev);
        return true;

      case "/clear":
        setMessages([{ id: genId(), role: "system", content: "History cleared." }]);
        setChatHistory([]);
        setShowHelp(false);
        return true;

      case "/exit":
      case "/quit":
        await saveSettings({ provider, model, agentMode });
        exit();
        return true;

      case "/agent":
        setAgentMode(prev => {
          const next = !prev;
          addMessage("system", `Agent mode ${next ? "ON" : "OFF"}.`);
          return next;
        });
        return true;

      case "/model":
        if (args[0]) {
          setModel(args[0]);
          await saveSettings({ model: args[0] });
          addMessage("system", `Model set to: ${args[0]}`);
        } else {
          addMessage("system", `Current model: ${model}. Usage: /model <model-id>`);
        }
        return true;

      case "/provider": {
        if (args[0]) {
          const newProvider = args[0].toLowerCase();
          const newModel = DEFAULT_MODELS[newProvider] ?? model;
          setProvider(newProvider);
          setModel(newModel);
          await saveSettings({ provider: newProvider, model: newModel });
          addMessage("system", `Provider: ${newProvider} • Model: ${newModel}`);
        } else {
          addMessage("system", `Current provider: ${provider}. Usage: /provider <name>`);
        }
        return true;
      }

      case "/tools":
        addMessage("system", `Enabled tools: ${enabledTools.join(", ") || "(none)"}`);
        return true;

      default:
        if (cmd.startsWith("/")) {
          addMessage("system", `Unknown command: ${cmd}. Type /help for commands.`);
          return true;
        }
        return false;
    }
  }, [provider, model, agentMode, enabledTools, addMessage, exit]);

  // ---------------------------------------------------------------------------
  // Chat submission
  // ---------------------------------------------------------------------------
  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    // Add to history (deduplicated)
    setInputHistory(prev => {
      const filtered = prev.filter(h => h !== trimmed);
      return [trimmed, ...filtered].slice(0, 100);
    });
    setHistoryIndex(-1);
    setHistoryDraft("");
    setInputValue("");
    setError(null);
    setShowHelp(false);

    // Handle commands
    if (trimmed.startsWith("/")) {
      await handleCommand(trimmed);
      return;
    }

    // Add user message to display
    await addMessage("user", trimmed);

    setStreaming(true);
    setStreamContent("");
    setStreamLabel("thinking");

    try {
      const prov = createProvider(providerRef.current);
      const ctx = new ProcessingContext({ jobId: crypto.randomUUID(), workspaceDir });
      const tools = buildTools();

      if (agentModeRef.current) {
        // --- Agent mode: use Agent class with a pre-defined task ---
        setStreamLabel("planning");
        const agent = new Agent({
          name: "chat-agent",
          objective: trimmed,
          provider: prov,
          model: modelRef.current,
          tools,
        });

        let assistantContent = "";
        for await (const msg of agent.execute(ctx)) {
          if (msg.type === "chunk") {
            const chunk = msg as { content?: string };
            assistantContent += chunk.content ?? "";
            setStreamContent(assistantContent);
            setStreamLabel("generating");
          } else if (msg.type === "tool_call_update") {
            const tc = msg as { name: string };
            setStreamLabel(`tool: ${tc.name}`);
          } else if (msg.type === "task_update") {
            const tu = msg as { event: string };
            setStreamLabel(`task: ${tu.event}`);
          } else if (msg.type === "planning_update") {
            const pu = msg as { content: string };
            setStreamLabel(`planning: ${pu.content.slice(0, 40)}`);
          } else if (msg.type === "step_result") {
            const sr = msg as { result: unknown; is_task_result: boolean };
            if (sr.is_task_result) {
              const result = typeof sr.result === "string"
                ? sr.result
                : JSON.stringify(sr.result, null, 2);
              assistantContent = result;
            }
          }
        }

        if (assistantContent) {
          await addMessage("assistant", assistantContent);
        }

      } else {
        // --- Regular chat mode ---
        let assistantContent = "";
        const updatedHistory = [...chatHistoryRef.current];

        await processChat({
          userInput: trimmed,
          messages: updatedHistory,
          model: modelRef.current,
          provider: prov,
          context: ctx,
          tools,
          callbacks: {
            onChunk: (text) => {
              assistantContent += text;
              setStreamContent(assistantContent);
              setStreamLabel("streaming");
            },
            onToolCall: (tc: ToolCall) => {
              setStreamLabel(`tool: ${tc.name}`);
            },
            onToolResult: (tc: ToolCall, result: unknown) => {
              const preview = typeof result === "string"
                ? result
                : JSON.stringify(result).slice(0, 100);
              addMessage("tool", preview, { toolName: tc.name });
            },
          },
        });

        setChatHistory(updatedHistory);

        if (assistantContent) {
          await addMessage("assistant", assistantContent);
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      await addMessage("system", `Error: ${msg}`);
    } finally {
      setStreaming(false);
      setStreamContent("");
      setStreamLabel("");
    }
  }, [handleCommand, addMessage, workspaceDir, enabledTools]);

  // ---------------------------------------------------------------------------
  // Keyboard: history navigation and tab completion
  // ---------------------------------------------------------------------------
  useInput((input, key) => {
    if (streaming) return; // block input while streaming

    if (key.upArrow) {
      setInputHistory(hist => {
        if (hist.length === 0) return hist;
        setHistoryIndex(prev => {
          if (prev === -1) setHistoryDraft(inputValue);
          const next = Math.min(prev + 1, hist.length - 1);
          setInputValue(hist[next] ?? "");
          return next;
        });
        return hist;
      });
      return;
    }

    if (key.downArrow) {
      setHistoryIndex(prev => {
        if (prev <= 0) {
          setInputValue(historyDraft);
          return -1;
        }
        const next = prev - 1;
        setInputHistory(hist => {
          setInputValue(hist[next] ?? "");
          return hist;
        });
        return next;
      });
      return;
    }

    // Tab: complete command names
    if (input === "\t" && inputValue.startsWith("/")) {
      const partial = inputValue.toLowerCase();
      const match = COMMAND_NAMES.find(c => c.startsWith(partial) && c !== partial);
      if (match) setInputValue(match + " ");
      return;
    }
  });

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <Box flexDirection="column">
      {/* Status bar */}
      <StatusBar
        provider={provider}
        model={model}
        agentMode={agentMode}
        msgCount={msgCount}
        streaming={streaming}
      />

      {/* Past messages — Static never re-renders past content */}
      <Static items={messages}>
        {(msg) => (
          <Box key={msg.id}>
            <ChatMessageItem msg={msg} />
          </Box>
        )}
      </Static>

      {/* Help panel (toggles) */}
      {showHelp && <HelpPanel />}

      {/* Live streaming area */}
      {streaming && (
        <Box flexDirection="column" marginLeft={2} marginTop={1}>
          <Box>
            <Text color="green" bold>  Assistant  </Text>
            <Text color="gray" dimColor>{"─".repeat(Math.max(0, (process.stdout.columns ?? 80) - 16))}</Text>
          </Box>
          <Box marginLeft={2}>
            <Text color="green" dimColor>{streamContent}</Text>
          </Box>
          <Box marginLeft={2} marginTop={1}>
            <Spinner type="dots" />
            <Text color="gray" dimColor> {streamLabel}</Text>
          </Box>
        </Box>
      )}

      {/* Error display */}
      {error && (
        <Box marginLeft={2} marginTop={1}>
          <Text color="red">✗ {error}</Text>
        </Box>
      )}

      {/* Input bar */}
      <Box marginTop={1}>
        <Text color="cyan" bold>{streaming ? "  " : "> "}</Text>
        {streaming
          ? <Text color="gray" dimColor>processing...</Text>
          : (
            <TextInput
              value={inputValue}
              onChange={setInputValue}
              onSubmit={handleSubmit}
              placeholder="Type a message or /help for commands"
            />
          )
        }
      </Box>

      {/* Hint line */}
      <Box>
        <Text color="gray" dimColor>  ↑↓ history  Tab: complete  /help: commands  Ctrl+C: exit</Text>
      </Box>
    </Box>
  );
}
