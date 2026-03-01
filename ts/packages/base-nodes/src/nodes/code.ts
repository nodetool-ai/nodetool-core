import { BaseNode } from "@nodetool/node-sdk";
import { spawn } from "node:child_process";

type RunResult = {
  stdout: string;
  stderr: string;
  exit_code: number;
};

async function runProcess(args: {
  cmd: string;
  argv?: string[];
  stdin?: string;
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs?: number;
}): Promise<RunResult> {
  const timeoutMs = args.timeoutMs ?? 30000;
  return new Promise((resolve, reject) => {
    const child = spawn(args.cmd, args.argv ?? [], {
      cwd: args.cwd,
      env: { ...process.env, ...(args.env ?? {}) },
      stdio: "pipe",
    });

    let stdout = "";
    let stderr = "";
    let killed = false;

    const timer = setTimeout(() => {
      killed = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.stdout.on("data", (d) => {
      stdout += String(d);
    });
    child.stderr.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (killed) {
        resolve({ stdout, stderr: `${stderr}\nProcess timed out`, exit_code: 124 });
        return;
      }
      resolve({ stdout, stderr, exit_code: code ?? 0 });
    });

    if (typeof args.stdin === "string" && args.stdin.length > 0) {
      child.stdin.write(args.stdin);
    }
    child.stdin.end();
  });
}

abstract class ScriptExecNode extends BaseNode {
  static readonly defaultCommand: string = "sh";
  static readonly defaultArgs: string[] = [];

  defaults() {
    return {
      script: "",
      cwd: "",
      env: {},
      timeout_ms: 30000,
    };
  }

  protected command(): { cmd: string; argv: string[] } {
    const cls = this.constructor as typeof ScriptExecNode;
    return {
      cmd: cls.defaultCommand,
      argv: [...cls.defaultArgs],
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const script = String(inputs.script ?? this._props.script ?? "");
    const cwdRaw = String(inputs.cwd ?? this._props.cwd ?? "");
    const cwd = cwdRaw || undefined;
    const timeoutMs = Number(inputs.timeout_ms ?? this._props.timeout_ms ?? 30000);
    const env = (inputs.env ?? this._props.env ?? {}) as Record<string, string>;
    const { cmd, argv } = this.command();
    const result = await runProcess({
      cmd,
      argv,
      stdin: script,
      cwd,
      env,
      timeoutMs,
    });
    return {
      ...result,
      output: result.stdout,
      success: result.exit_code === 0,
    };
  }
}

export class ExecutePythonNode extends ScriptExecNode {
  static readonly nodeType = "nodetool.code.ExecutePython";
  static readonly title = "Execute Python";
  static readonly description = "Execute Python script";
  static readonly defaultCommand = "python3";
  static readonly defaultArgs = ["-"];
}

export class ExecuteJavaScriptNode extends ScriptExecNode {
  static readonly nodeType = "nodetool.code.ExecuteJavaScript";
  static readonly title = "Execute JavaScript";
  static readonly description = "Execute JavaScript script";
  static readonly defaultCommand = "node";
  static readonly defaultArgs = ["-"];
}

export class ExecuteBashNode extends ScriptExecNode {
  static readonly nodeType = "nodetool.code.ExecuteBash";
  static readonly title = "Execute Bash";
  static readonly description = "Execute Bash script";
  static readonly defaultCommand = "bash";
  static readonly defaultArgs = ["-s"];
}

export class ExecuteRubyNode extends ScriptExecNode {
  static readonly nodeType = "nodetool.code.ExecuteRuby";
  static readonly title = "Execute Ruby";
  static readonly description = "Execute Ruby script";
  static readonly defaultCommand = "ruby";
  static readonly defaultArgs = ["-"];
}

export class ExecuteLuaNode extends ScriptExecNode {
  static readonly nodeType = "nodetool.code.ExecuteLua";
  static readonly title = "Execute Lua";
  static readonly description = "Execute Lua script";
  static readonly defaultCommand = "lua";
  static readonly defaultArgs = ["-"];
}

export class ExecuteCommandNode extends BaseNode {
  static readonly nodeType = "nodetool.code.ExecuteCommand";
  static readonly title = "Execute Command";
  static readonly description = "Execute shell command";

  defaults() {
    return {
      command: "",
      cwd: "",
      env: {},
      timeout_ms: 30000,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const command = String(inputs.command ?? this._props.command ?? "");
    const cwdRaw = String(inputs.cwd ?? this._props.cwd ?? "");
    const cwd = cwdRaw || undefined;
    const timeoutMs = Number(inputs.timeout_ms ?? this._props.timeout_ms ?? 30000);
    const env = (inputs.env ?? this._props.env ?? {}) as Record<string, string>;

    const result = await runProcess({
      cmd: "sh",
      argv: ["-c", command],
      cwd,
      env,
      timeoutMs,
    });

    return {
      ...result,
      output: result.stdout,
      success: result.exit_code === 0,
    };
  }
}

abstract class RunCommandNode extends BaseNode {
  static readonly runtime: string = "sh";
  static readonly runtimeArgs: string[] = ["-c"];

  defaults() {
    return {
      command: "",
      cwd: "",
      env: {},
      timeout_ms: 30000,
    };
  }

  protected runtimeConfig(): { cmd: string; args: string[] } {
    const cls = this.constructor as typeof RunCommandNode;
    return { cmd: cls.runtime, args: [...cls.runtimeArgs] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const command = String(inputs.command ?? this._props.command ?? "");
    const cwdRaw = String(inputs.cwd ?? this._props.cwd ?? "");
    const cwd = cwdRaw || undefined;
    const timeoutMs = Number(inputs.timeout_ms ?? this._props.timeout_ms ?? 30000);
    const env = (inputs.env ?? this._props.env ?? {}) as Record<string, string>;
    const { cmd, args } = this.runtimeConfig();
    const result = await runProcess({
      cmd,
      argv: [...args, command],
      cwd,
      env,
      timeoutMs,
    });
    return {
      ...result,
      output: result.stdout,
      success: result.exit_code === 0,
    };
  }
}

export class RunPythonCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunPythonCommand";
  static readonly title = "Run Python Command";
  static readonly description = "Run python -c command";
  static readonly runtime = "python3";
  static readonly runtimeArgs = ["-c"];
}

export class RunJavaScriptCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunJavaScriptCommand";
  static readonly title = "Run JavaScript Command";
  static readonly description = "Run node -e command";
  static readonly runtime = "node";
  static readonly runtimeArgs = ["-e"];
}

export class RunBashCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunBashCommand";
  static readonly title = "Run Bash Command";
  static readonly description = "Run bash -c command";
  static readonly runtime = "bash";
  static readonly runtimeArgs = ["-c"];
}

export class RunRubyCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunRubyCommand";
  static readonly title = "Run Ruby Command";
  static readonly description = "Run ruby -e command";
  static readonly runtime = "ruby";
  static readonly runtimeArgs = ["-e"];
}

export class RunLuaCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunLuaCommand";
  static readonly title = "Run Lua Command";
  static readonly description = "Run lua -e command";
  static readonly runtime = "lua";
  static readonly runtimeArgs = ["-e"];
}

export class RunLuaCommandDockerNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunLuaCommandDocker";
  static readonly title = "Run Lua Command Docker";
  static readonly description = "Docker variant alias of RunLuaCommand";
  static readonly runtime = "lua";
  static readonly runtimeArgs = ["-e"];
}

export class RunShellCommandNode extends RunCommandNode {
  static readonly nodeType = "nodetool.code.RunShellCommand";
  static readonly title = "Run Shell Command";
  static readonly description = "Run shell command with sh -c";
  static readonly runtime = "sh";
  static readonly runtimeArgs = ["-c"];
}

export const CODE_NODES = [
  ExecutePythonNode,
  ExecuteJavaScriptNode,
  ExecuteBashNode,
  ExecuteRubyNode,
  ExecuteLuaNode,
  ExecuteCommandNode,
  RunPythonCommandNode,
  RunJavaScriptCommandNode,
  RunBashCommandNode,
  RunRubyCommandNode,
  RunLuaCommandNode,
  RunLuaCommandDockerNode,
  RunShellCommandNode,
] as const;
