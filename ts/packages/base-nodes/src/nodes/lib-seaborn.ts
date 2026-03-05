import { BaseNode } from "@nodetool/node-sdk";
import type { NodeClass } from "@nodetool/node-sdk";

export class ChartRendererLibNode extends BaseNode {
  static readonly nodeType = "lib.seaborn.ChartRenderer";
  static readonly title = "Chart Renderer";
  static readonly description =
    "Node responsible for rendering chart configurations into image format using seaborn.";

  defaults() {
    return {
      chart_config: {
        title: "",
        x_label: "",
        y_label: "",
        data: { series: [] },
      },
      width: 640,
      height: 480,
      data: { columns: [], data: [] },
    };
  }

  async process(
    inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    const config = (inputs.chart_config ?? this._props.chart_config ?? {}) as Record<string, unknown>;
    const width = Number(inputs.width ?? this._props.width ?? 640);
    const height = Number(inputs.height ?? this._props.height ?? 480);
    const dataRef = (inputs.data ?? this._props.data ?? {}) as Record<string, unknown>;

    const columns = (dataRef.columns ?? []) as Array<Record<string, unknown>>;
    const rows = (dataRef.data ?? []) as unknown[][];

    if (!rows.length) {
      throw new Error("Data is required for rendering the chart.");
    }

    const { ChartJSNodeCanvas } = await import("chartjs-node-canvas");

    const configData = (config.data ?? {}) as Record<string, unknown>;
    const series = (configData.series ?? []) as Array<Record<string, unknown>>;

    // Map column names
    const colNames = columns.map((c) => String(c.name ?? c));

    // Determine chart type from first series (default to bar)
    const plotTypeMap: Record<string, string> = {
      scatter: "scatter",
      line: "line",
      barplot: "bar",
      histplot: "bar",
      boxplot: "bar",
      pointplot: "line",
      countplot: "bar",
    };

    const firstSeries = series[0] ?? {};
    const plotType = String(firstSeries.plot_type ?? "barplot").toLowerCase();
    const chartType = plotTypeMap[plotType] ?? "bar";

    // Extract x and y column indices
    const xCol = String(firstSeries.x ?? colNames[0] ?? "");
    const yCol = String(firstSeries.y ?? colNames[1] ?? "");
    const xIdx = colNames.indexOf(xCol);
    const yIdx = colNames.indexOf(yCol);

    const labels = xIdx >= 0 ? rows.map((r) => String(r[xIdx])) : rows.map((_, i) => String(i));
    const values = yIdx >= 0 ? rows.map((r) => Number(r[yIdx])) : [];

    const datasets = series.length > 0
      ? series.map((s) => {
          const sYCol = String(s.y ?? yCol);
          const sYIdx = colNames.indexOf(sYCol);
          const data = sYIdx >= 0 ? rows.map((r) => Number(r[sYIdx])) : values;
          return {
            label: sYCol,
            data,
            backgroundColor: s.color ? String(s.color) : undefined,
          };
        })
      : [{ label: yCol, data: values }];

    const chartConfig = {
      type: chartType as "bar" | "line" | "scatter",
      data: { labels, datasets },
      options: {
        responsive: false,
        plugins: {
          title: config.title
            ? { display: true, text: String(config.title) }
            : undefined,
        },
        scales: {
          x: config.x_label
            ? { title: { display: true, text: String(config.x_label) } }
            : undefined,
          y: config.y_label
            ? { title: { display: true, text: String(config.y_label) } }
            : undefined,
        },
      },
    };

    const canvas = new ChartJSNodeCanvas({ width, height });
    const buffer = await canvas.renderToBuffer(chartConfig as Parameters<typeof canvas.renderToBuffer>[0]);
    const data = Buffer.from(buffer).toString("base64");

    return { output: { type: "image", data } };
  }
}

export const LIB_SEABORN_NODES: readonly NodeClass[] = [
  ChartRendererLibNode,
] as const;
