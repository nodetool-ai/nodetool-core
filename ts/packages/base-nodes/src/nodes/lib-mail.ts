import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

export class SendEmailLibNode extends BaseNode {
  static readonly nodeType = "lib.mail.SendEmail";
  static readonly title = "Send Email";
  static readonly description = "Send a plain text email via SMTP.";

  defaults() {
    return {
      smtp_server: "smtp.gmail.com",
      smtp_port: 587,
      username: "",
      password: "",
      from_address: "",
      to_address: "",
      subject: "",
      body: "",
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const smtpServer = String(inputs.smtp_server ?? this._props.smtp_server ?? "smtp.gmail.com");
    const smtpPort = Number(inputs.smtp_port ?? this._props.smtp_port ?? 587);
    const username = String(inputs.username ?? this._props.username ?? "");
    const password = String(inputs.password ?? this._props.password ?? "");
    const fromAddress = String(inputs.from_address ?? this._props.from_address ?? "");
    const toAddress = String(inputs.to_address ?? this._props.to_address ?? "");
    const subject = String(inputs.subject ?? this._props.subject ?? "");
    const body = String(inputs.body ?? this._props.body ?? "");

    if (!toAddress) throw new Error("Recipient email address is required");

    const nodemailer = (await import("nodemailer")).default;
    const transporter = nodemailer.createTransport({
      host: smtpServer,
      port: smtpPort,
      secure: false,
      auth: username ? { user: username, pass: password } : undefined,
    });

    await transporter.sendMail({
      from: fromAddress || username,
      to: toAddress,
      subject,
      text: body,
    });

    return { output: true };
  }
}

export class GmailSearchLibNode extends BaseNode {
  static readonly nodeType = "lib.mail.GmailSearch";
  static readonly title = "Gmail Search";
  static readonly description =
    "Searches Gmail using Gmail-specific search operators and yields matching emails.";

  defaults() {
    return {
      from_address: "",
      to_address: "",
      subject: "",
      body: "",
      date_filter: "SINCE_ONE_DAY",
      keywords: "",
      folder: "INBOX",
      text: "",
      max_results: 50,
    };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    throw new Error(
      "GmailSearch requires Google OAuth2/IMAP credentials which are not available in the TypeScript runtime. " +
      "Configure GOOGLE_MAIL_USER and GOOGLE_APP_PASSWORD environment variables and use the Python runtime."
    );
  }
}

export class AddLabelLibNode extends BaseNode {
  static readonly nodeType = "lib.mail.AddLabel";
  static readonly title = "Add Label";
  static readonly description = "Adds a label to a Gmail message.";

  defaults() {
    return { message_id: "", label: "" };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    throw new Error(
      "AddLabel requires Google OAuth2/IMAP credentials which are not available in the TypeScript runtime. " +
      "Configure GOOGLE_MAIL_USER and GOOGLE_APP_PASSWORD environment variables and use the Python runtime."
    );
  }
}

export class MoveToArchiveLibNode extends BaseNode {
  static readonly nodeType = "lib.mail.MoveToArchive";
  static readonly title = "Move To Archive";
  static readonly description = "Moves specified emails to Gmail archive.";

  defaults() {
    return { message_id: "" };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    throw new Error(
      "MoveToArchive requires Google OAuth2/IMAP credentials which are not available in the TypeScript runtime. " +
      "Configure GOOGLE_MAIL_USER and GOOGLE_APP_PASSWORD environment variables and use the Python runtime."
    );
  }
}

export const LIB_MAIL_NODES: readonly NodeClass[] = [
  SendEmailLibNode,
  GmailSearchLibNode,
  AddLabelLibNode,
  MoveToArchiveLibNode,
] as const;
