import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

// ── Discord Nodes ───────────────────────────────────────────────────────────

export class DiscordBotTrigger extends BaseNode {
  static readonly nodeType = "messaging.discord.DiscordBotTrigger";
  static readonly title = "Discord Bot Trigger";
  static readonly description =
    "Trigger node that listens for Discord messages from a bot account. " +
    "Connects to Discord using a bot token and emits events for incoming messages.";

  defaults() {
    return {
      token: "",
      channel_id: "",
      allow_bot_messages: false,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const secrets = (inputs._secrets as Record<string, string>) ?? {};
    const token =
      String(inputs.token ?? this._props.token ?? "") ||
      secrets.DISCORD_BOT_TOKEN ||
      "";
    const channelId = String(
      inputs.channel_id ?? this._props.channel_id ?? ""
    );
    const allowBotMessages = Boolean(
      inputs.allow_bot_messages ?? this._props.allow_bot_messages ?? false
    );

    if (!token) {
      throw new Error("Discord bot token is required");
    }

    // Validate the bot token by fetching the bot user info
    const resp = await fetch("https://discord.com/api/v10/users/@me", {
      headers: { Authorization: `Bot ${token}` },
    });

    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`Discord token validation failed (${resp.status}): ${body}`);
    }

    const botUser = (await resp.json()) as Record<string, unknown>;

    return {
      status: "configured",
      bot_id: botUser.id,
      bot_username: botUser.username,
      channel_id: channelId,
      allow_bot_messages: allowBotMessages,
    };
  }
}

export class DiscordSendMessage extends BaseNode {
  static readonly nodeType = "messaging.discord.DiscordSendMessage";
  static readonly title = "Discord Send Message";
  static readonly description =
    "Sends a message to a Discord channel using a bot token.";

  defaults() {
    return {
      token: "",
      channel_id: "",
      content: "",
      tts: false,
      embeds: [],
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const secrets = (inputs._secrets as Record<string, string>) ?? {};
    const token =
      String(inputs.token ?? this._props.token ?? "") ||
      secrets.DISCORD_BOT_TOKEN ||
      "";
    const channelId = String(
      inputs.channel_id ?? this._props.channel_id ?? ""
    );
    const content = String(inputs.content ?? this._props.content ?? "");
    const tts = Boolean(inputs.tts ?? this._props.tts ?? false);
    const embeds = (inputs.embeds ?? this._props.embeds ?? []) as unknown[];

    if (!token) {
      throw new Error("Discord bot token is required");
    }
    if (!channelId) {
      throw new Error("Discord channel ID is required");
    }

    const payload: Record<string, unknown> = {
      content,
      tts,
    };
    if (Array.isArray(embeds) && embeds.length > 0) {
      payload.embeds = embeds;
    }

    const resp = await fetch(
      `https://discord.com/api/v10/channels/${channelId}/messages`,
      {
        method: "POST",
        headers: {
          Authorization: `Bot ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      }
    );

    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(
        `Discord sendMessage failed (${resp.status}): ${body}`
      );
    }

    const msg = (await resp.json()) as Record<string, unknown>;
    return {
      message_id: msg.id,
    };
  }
}

// ── Telegram Nodes ──────────────────────────────────────────────────────────

export class TelegramBotTrigger extends BaseNode {
  static readonly nodeType = "messaging.telegram.TelegramBotTrigger";
  static readonly title = "Telegram Bot Trigger";
  static readonly description =
    "Trigger node that listens for Telegram messages using long polling. " +
    "Connects to Telegram using a bot token and emits events for incoming messages.";

  defaults() {
    return {
      token: "",
      chat_id: 0,
      allow_bot_messages: false,
      include_edited_messages: false,
      poll_timeout_seconds: 30,
      poll_interval_seconds: 0.2,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const secrets = (inputs._secrets as Record<string, string>) ?? {};
    const token =
      String(inputs.token ?? this._props.token ?? "") ||
      secrets.TELEGRAM_BOT_TOKEN ||
      "";
    const chatId = Number(inputs.chat_id ?? this._props.chat_id ?? 0);
    const allowBotMessages = Boolean(
      inputs.allow_bot_messages ?? this._props.allow_bot_messages ?? false
    );
    const includeEditedMessages = Boolean(
      inputs.include_edited_messages ??
        this._props.include_edited_messages ??
        false
    );

    if (!token) {
      throw new Error("Telegram bot token is required");
    }

    // Validate the bot token by calling getMe
    const resp = await fetch(
      `https://api.telegram.org/bot${token}/getMe`
    );

    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(
        `Telegram token validation failed (${resp.status}): ${body}`
      );
    }

    const data = (await resp.json()) as Record<string, unknown>;
    if (!data.ok) {
      throw new Error(`Telegram getMe failed: ${JSON.stringify(data)}`);
    }

    const result = data.result as Record<string, unknown>;
    return {
      status: "configured",
      bot_id: result.id,
      bot_username: result.username,
      chat_id: chatId || null,
      allow_bot_messages: allowBotMessages,
      include_edited_messages: includeEditedMessages,
    };
  }
}

export class TelegramSendMessage extends BaseNode {
  static readonly nodeType = "messaging.telegram.TelegramSendMessage";
  static readonly title = "Telegram Send Message";
  static readonly description =
    "Sends a message to a Telegram chat using a bot token.";

  defaults() {
    return {
      token: "",
      chat_id: 0,
      text: "",
      parse_mode: "",
      disable_web_page_preview: false,
      disable_notification: false,
      reply_to_message_id: 0,
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const secrets = (inputs._secrets as Record<string, string>) ?? {};
    const token =
      String(inputs.token ?? this._props.token ?? "") ||
      secrets.TELEGRAM_BOT_TOKEN ||
      "";
    const chatId = Number(inputs.chat_id ?? this._props.chat_id ?? 0);
    const text = String(inputs.text ?? this._props.text ?? "");
    const parseMode = String(
      inputs.parse_mode ?? this._props.parse_mode ?? ""
    );
    const disableWebPagePreview = Boolean(
      inputs.disable_web_page_preview ??
        this._props.disable_web_page_preview ??
        false
    );
    const disableNotification = Boolean(
      inputs.disable_notification ??
        this._props.disable_notification ??
        false
    );
    const replyToMessageId = Number(
      inputs.reply_to_message_id ?? this._props.reply_to_message_id ?? 0
    );

    if (!token) {
      throw new Error("Telegram bot token is required");
    }
    if (!chatId) {
      throw new Error("Telegram chat ID is required");
    }

    const payload: Record<string, unknown> = {
      chat_id: chatId,
      text,
      disable_web_page_preview: disableWebPagePreview,
      disable_notification: disableNotification,
    };

    if (parseMode) {
      payload.parse_mode = parseMode;
    }
    if (replyToMessageId) {
      payload.reply_to_message_id = replyToMessageId;
    }

    const resp = await fetch(
      `https://api.telegram.org/bot${token}/sendMessage`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    const data = (await resp.json()) as Record<string, unknown>;

    if (!data.ok) {
      throw new Error(`Telegram sendMessage failed: ${JSON.stringify(data)}`);
    }

    const result = data.result as Record<string, unknown>;
    const chat = (result.chat as Record<string, unknown>) ?? {};

    return {
      message_id: result.message_id,
      date: result.date,
      chat_id: chat.id,
    };
  }
}

// ── Export ───────────────────────────────────────────────────────────────────

export const MESSAGING_NODES: readonly NodeClass[] = [
  DiscordBotTrigger,
  DiscordSendMessage,
  TelegramBotTrigger,
  TelegramSendMessage,
];
