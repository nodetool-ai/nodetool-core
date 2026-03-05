import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  DiscordBotTrigger,
  DiscordSendMessage,
  TelegramBotTrigger,
  TelegramSendMessage,
} from "../src/nodes/messaging.js";

const originalFetch = globalThis.fetch;
let mockFetch: ReturnType<typeof vi.fn>;

beforeEach(() => {
  mockFetch = vi.fn();
  globalThis.fetch = mockFetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

// ── DiscordBotTrigger ──────────────────────────────────────────────────────

describe("DiscordBotTrigger", () => {
  it("returns configured status with bot info", async () => {
    const node = new DiscordBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ id: "123", username: "TestBot" })
    );
    const result = await node.process({
      token: "bot-token",
      channel_id: "456",
    });
    expect(result.status).toBe("configured");
    expect(result.bot_id).toBe("123");
    expect(result.bot_username).toBe("TestBot");
    expect(result.channel_id).toBe("456");
    expect(result.allow_bot_messages).toBe(false);
  });

  it("uses token from _secrets", async () => {
    const node = new DiscordBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ id: "1", username: "Bot" })
    );
    const result = await node.process({
      _secrets: { DISCORD_BOT_TOKEN: "secret-token" },
    });
    expect(result.status).toBe("configured");
    const headers = mockFetch.mock.calls[0][1]?.headers ?? {};
    // The fetch is called with just the URL and headers object
    expect(mockFetch.mock.calls[0][0]).toContain("discord.com");
  });

  it("throws when no token", async () => {
    const node = new DiscordBotTrigger();
    await expect(node.process({})).rejects.toThrow(
      "Discord bot token is required"
    );
  });

  it("throws on API validation failure", async () => {
    const node = new DiscordBotTrigger();
    mockFetch.mockResolvedValueOnce(jsonResponse("Unauthorized", 401));
    await expect(
      node.process({ token: "bad-token" })
    ).rejects.toThrow("Discord token validation failed (401)");
  });
});

// ── DiscordSendMessage ─────────────────────────────────────────────────────

describe("DiscordSendMessage", () => {
  it("sends message and returns message_id", async () => {
    const node = new DiscordSendMessage();
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "msg123" }));
    const result = await node.process({
      token: "bot-token",
      channel_id: "ch456",
      content: "Hello!",
    });
    expect(result.message_id).toBe("msg123");
  });

  it("includes embeds when provided", async () => {
    const node = new DiscordSendMessage();
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "msg2" }));
    await node.process({
      token: "bot-token",
      channel_id: "ch1",
      content: "text",
      embeds: [{ title: "embed" }],
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.embeds).toEqual([{ title: "embed" }]);
  });

  it("throws when no token", async () => {
    const node = new DiscordSendMessage();
    await expect(
      node.process({ channel_id: "ch1", content: "hi" })
    ).rejects.toThrow("Discord bot token is required");
  });

  it("throws when no channel_id", async () => {
    const node = new DiscordSendMessage();
    await expect(
      node.process({ token: "tok", content: "hi" })
    ).rejects.toThrow("Discord channel ID is required");
  });

  it("throws on API error", async () => {
    const node = new DiscordSendMessage();
    mockFetch.mockResolvedValueOnce(jsonResponse("forbidden", 403));
    await expect(
      node.process({
        token: "tok",
        channel_id: "ch1",
        content: "hi",
      })
    ).rejects.toThrow("Discord sendMessage failed (403)");
  });

  it("uses token from _secrets", async () => {
    const node = new DiscordSendMessage();
    mockFetch.mockResolvedValueOnce(jsonResponse({ id: "msg3" }));
    const result = await node.process({
      _secrets: { DISCORD_BOT_TOKEN: "secret-tok" },
      channel_id: "ch1",
      content: "hi",
    });
    expect(result.message_id).toBe("msg3");
  });
});

// ── TelegramBotTrigger ─────────────────────────────────────────────────────

describe("TelegramBotTrigger", () => {
  it("returns configured status with bot info", async () => {
    const node = new TelegramBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { id: 789, username: "test_bot" },
      })
    );
    const result = await node.process({
      token: "tg-token",
      chat_id: 12345,
    });
    expect(result.status).toBe("configured");
    expect(result.bot_id).toBe(789);
    expect(result.bot_username).toBe("test_bot");
    expect(result.chat_id).toBe(12345);
  });

  it("returns null chat_id when 0", async () => {
    const node = new TelegramBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { id: 1, username: "bot" },
      })
    );
    const result = await node.process({ token: "tok" });
    expect(result.chat_id).toBeNull();
  });

  it("uses token from _secrets", async () => {
    const node = new TelegramBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { id: 1, username: "bot" },
      })
    );
    await node.process({
      _secrets: { TELEGRAM_BOT_TOKEN: "secret-tg" },
    });
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("botsecret-tg");
  });

  it("throws when no token", async () => {
    const node = new TelegramBotTrigger();
    await expect(node.process({})).rejects.toThrow(
      "Telegram bot token is required"
    );
  });

  it("throws on HTTP validation failure", async () => {
    const node = new TelegramBotTrigger();
    mockFetch.mockResolvedValueOnce(jsonResponse("bad token", 401));
    await expect(
      node.process({ token: "bad" })
    ).rejects.toThrow("Telegram token validation failed (401)");
  });

  it("throws when getMe returns ok: false", async () => {
    const node = new TelegramBotTrigger();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ ok: false, description: "Not found" })
    );
    await expect(
      node.process({ token: "tok" })
    ).rejects.toThrow("Telegram getMe failed");
  });
});

// ── TelegramSendMessage ────────────────────────────────────────────────────

describe("TelegramSendMessage", () => {
  it("sends message and returns result", async () => {
    const node = new TelegramSendMessage();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: {
          message_id: 42,
          date: 1234567890,
          chat: { id: 100 },
        },
      })
    );
    const result = await node.process({
      token: "tg-token",
      chat_id: 100,
      text: "Hello!",
    });
    expect(result.message_id).toBe(42);
    expect(result.date).toBe(1234567890);
    expect(result.chat_id).toBe(100);
  });

  it("includes parse_mode when set", async () => {
    const node = new TelegramSendMessage();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { message_id: 1, date: 0, chat: { id: 1 } },
      })
    );
    await node.process({
      token: "tok",
      chat_id: 1,
      text: "hi",
      parse_mode: "HTML",
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.parse_mode).toBe("HTML");
  });

  it("includes reply_to_message_id when set", async () => {
    const node = new TelegramSendMessage();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { message_id: 2, date: 0, chat: { id: 1 } },
      })
    );
    await node.process({
      token: "tok",
      chat_id: 1,
      text: "reply",
      reply_to_message_id: 99,
    });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.reply_to_message_id).toBe(99);
  });

  it("throws when no token", async () => {
    const node = new TelegramSendMessage();
    await expect(
      node.process({ chat_id: 1, text: "hi" })
    ).rejects.toThrow("Telegram bot token is required");
  });

  it("throws when no chat_id", async () => {
    const node = new TelegramSendMessage();
    await expect(
      node.process({ token: "tok", text: "hi" })
    ).rejects.toThrow("Telegram chat ID is required");
  });

  it("throws when API returns ok: false", async () => {
    const node = new TelegramSendMessage();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: false,
        description: "Bad Request: chat not found",
      })
    );
    await expect(
      node.process({ token: "tok", chat_id: 999, text: "hi" })
    ).rejects.toThrow("Telegram sendMessage failed");
  });

  it("uses token from _secrets", async () => {
    const node = new TelegramSendMessage();
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        ok: true,
        result: { message_id: 5, date: 0, chat: { id: 1 } },
      })
    );
    await node.process({
      _secrets: { TELEGRAM_BOT_TOKEN: "secret" },
      chat_id: 1,
      text: "hi",
    });
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("botsecret/sendMessage");
  });
});

// ── Defaults coverage ────────────────────────────────────────────────────

describe("Node defaults coverage", () => {
  it("DiscordBotTrigger defaults", () => {
    const node = new DiscordBotTrigger();
    const d = node.defaults();
    expect(d.token).toBe("");
    expect(d.channel_id).toBe("");
    expect(d.allow_bot_messages).toBe(false);
  });

  it("DiscordSendMessage defaults", () => {
    const node = new DiscordSendMessage();
    const d = node.defaults();
    expect(d.token).toBe("");
    expect(d.channel_id).toBe("");
    expect(d.content).toBe("");
    expect(d.tts).toBe(false);
    expect(d.embeds).toEqual([]);
  });

  it("TelegramBotTrigger defaults", () => {
    const node = new TelegramBotTrigger();
    const d = node.defaults();
    expect(d.token).toBe("");
    expect(d.chat_id).toBe(0);
    expect(d.allow_bot_messages).toBe(false);
    expect(d.include_edited_messages).toBe(false);
    expect(d.poll_timeout_seconds).toBe(30);
    expect(d.poll_interval_seconds).toBe(0.2);
  });

  it("TelegramSendMessage defaults", () => {
    const node = new TelegramSendMessage();
    const d = node.defaults();
    expect(d.token).toBe("");
    expect(d.chat_id).toBe(0);
    expect(d.text).toBe("");
    expect(d.parse_mode).toBe("");
    expect(d.disable_web_page_preview).toBe(false);
    expect(d.disable_notification).toBe(false);
    expect(d.reply_to_message_id).toBe(0);
  });
});
