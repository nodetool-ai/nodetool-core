declare module "marked-terminal" {
  import type { MarkedExtension } from "marked";
  interface TerminalRendererOptions {
    code?: boolean;
    blockquote?: boolean;
    html?: boolean;
    heading?: boolean;
    firstHeading?: boolean;
    hr?: boolean;
    listitem?: boolean;
    table?: boolean;
    paragraph?: boolean;
    strong?: boolean;
    em?: boolean;
    codespan?: boolean;
    del?: boolean;
    link?: boolean;
    href?: boolean;
    tableOptions?: object;
    unescape?: boolean;
    emoji?: boolean;
    width?: number;
    showSectionPrefix?: boolean;
    reflowText?: boolean;
    tab?: number;
  }
  class TerminalRenderer {
    constructor(options?: TerminalRendererOptions);
  }
  export default TerminalRenderer;
}
