## 2024-05-26 - Recursive DFS limit in Deep Graphs
**Learning:** Python's default recursion limit (usually 1000) causes `RecursionError` in workflow systems when parsing deeply nested or long linear graph definitions using recursive algorithms like recursive DFS. For example, validating `control_edges` on a graph with 5000 linearly connected nodes will crash.
**Action:** When working with graph validation or traversal in workflow codebases, always prefer iterative algorithms over recursive ones (e.g., Kahn's Algorithm for topological sorting or iterative DFS using an explicit stack).
