import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// We need to test the cli module which auto-executes main() on import.
// We mock the shadow module and process.exit to prevent actual exits.

describe("cli - parseArgs and splitCommand", () => {
  let originalArgv: string[];
  let stdoutWrite: ReturnType<typeof vi.spyOn>;
  let stderrWrite: ReturnType<typeof vi.spyOn>;
  let exitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    originalArgv = [...process.argv];
    stdoutWrite = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    stderrWrite = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    // Mock process.exit to NOT throw - just record the call
    exitSpy = vi.spyOn(process, "exit").mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    process.argv = originalArgv;
    stdoutWrite.mockRestore();
    stderrWrite.mockRestore();
    exitSpy.mockRestore();
    vi.restoreAllMocks();
  });

  async function importCli(args: string[], mockComparison?: any) {
    process.argv = ["node", "cli.js", ...args];
    vi.resetModules();
    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: vi.fn().mockResolvedValue(
        mockComparison ?? {
          python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
          ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
          drifts: [],
        }
      ),
    }));

    await import("../src/cli.js");
    // Allow the main() promise to settle
    await new Promise((r) => setTimeout(r, 150));
  }

  it("parses --python and --ts args and calls runShadowComparison", async () => {
    process.argv = ["node", "cli.js", "--python", "python run.py", "--ts", "node run.js"];
    vi.resetModules();

    const mockFn = vi.fn().mockResolvedValue({
      python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      drifts: [],
    });

    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: mockFn,
    }));

    await import("../src/cli.js");
    await new Promise((r) => setTimeout(r, 150));

    expect(mockFn).toHaveBeenCalledWith(
      { cmd: "python", args: ["run.py"], cwd: undefined },
      { cmd: "node", args: ["run.js"], cwd: undefined }
    );
  });

  it("parses --cwd argument", async () => {
    process.argv = [
      "node", "cli.js",
      "--python", "python run.py",
      "--ts", "node run.js",
      "--cwd", "/tmp/test",
    ];
    vi.resetModules();

    const mockFn = vi.fn().mockResolvedValue({
      python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      drifts: [],
    });

    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: mockFn,
    }));

    await import("../src/cli.js");
    await new Promise((r) => setTimeout(r, 150));

    expect(mockFn).toHaveBeenCalledWith(
      { cmd: "python", args: ["run.py"], cwd: "/tmp/test" },
      { cmd: "node", args: ["run.js"], cwd: "/tmp/test" }
    );
  });

  it("exits with code 0 when no drifts", async () => {
    await importCli(["--python", "echo hi", "--ts", "echo hi"]);
    expect(exitSpy).toHaveBeenCalledWith(0);
  });

  it("exits with code 2 when drifts exist", async () => {
    await importCli(["--python", "echo hi", "--ts", "echo hi"], {
      python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      drifts: [{ category: "output_drift", index: 0, message: "test drift" }],
    });

    expect(exitSpy).toHaveBeenCalledWith(2);
  });

  it("writes JSON report to stdout", async () => {
    await importCli(["--python", "echo hi", "--ts", "echo hi"]);

    expect(stdoutWrite).toHaveBeenCalled();
    const output = stdoutWrite.mock.calls[0]?.[0] as string;
    const report = JSON.parse(output);
    expect(report).toHaveProperty("pythonExit", 0);
    expect(report).toHaveProperty("tsExit", 0);
    expect(report).toHaveProperty("driftCount", 0);
    expect(report).toHaveProperty("drifts");
  });

  it("exits with code 1 and writes to stderr on missing args", async () => {
    await importCli([]);

    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(stderrWrite).toHaveBeenCalled();
    const errOutput = stderrWrite.mock.calls[0]?.[0] as string;
    expect(errOutput).toContain("Usage:");
  });

  it("exits with code 1 when only --python is provided", async () => {
    await importCli(["--python", "echo hi"]);

    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(stderrWrite).toHaveBeenCalled();
  });

  it("exits with code 1 when runShadowComparison rejects", async () => {
    process.argv = ["node", "cli.js", "--python", "echo hi", "--ts", "echo hi"];
    vi.resetModules();
    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: vi.fn().mockRejectedValue(new Error("spawn failed")),
    }));

    await import("../src/cli.js");
    await new Promise((r) => setTimeout(r, 150));

    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(stderrWrite).toHaveBeenCalled();
    const errOutput = stderrWrite.mock.calls[0]?.[0] as string;
    expect(errOutput).toContain("spawn failed");
  });

  it("exits with code 1 when command string is empty", async () => {
    // A whitespace-only --python value passes parseArgs but fails splitCommand
    process.argv = ["node", "cli.js", "--python", "  ", "--ts", "echo hi"];
    vi.resetModules();
    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: vi.fn().mockResolvedValue({
        python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
        ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
        drifts: [],
      }),
    }));

    await import("../src/cli.js");
    await new Promise((r) => setTimeout(r, 150));

    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(stderrWrite).toHaveBeenCalled();
    const errOutput = stderrWrite.mock.calls[0]?.[0] as string;
    expect(errOutput).toContain("Command is empty");
  });

  it("handles splitCommand with multi-space separated commands", async () => {
    process.argv = ["node", "cli.js", "--python", "python  -u  run.py", "--ts", "node run.js"];
    vi.resetModules();

    const mockFn = vi.fn().mockResolvedValue({
      python: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      ts: { exitCode: 0, stdoutLines: [], stderr: "", messages: [] },
      drifts: [],
    });

    vi.doMock("../src/shadow.js", () => ({
      runShadowComparison: mockFn,
    }));

    await import("../src/cli.js");
    await new Promise((r) => setTimeout(r, 150));

    // splitCommand filters empty parts from splitting on spaces
    expect(mockFn).toHaveBeenCalledWith(
      { cmd: "python", args: ["-u", "run.py"], cwd: undefined },
      { cmd: "node", args: ["run.js"], cwd: undefined }
    );
  });
});
