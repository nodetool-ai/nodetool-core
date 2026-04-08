## 2026-04-08 - String Concatenation and List Extensions Optimization
**Learning:** In Python, string concatenations (`+=`) in loops create O(n^2) performance bottlenecks because strings are immutable, requiring intermediate copies. A similar, though less severe, reallocation impact happens with `+=` on lists compared to `.extend()`.
**Action:** Use the pattern `list.append()` followed by `"".join(list)` when dynamically building large strings iteratively. Similarly, prefer `.extend()` when merging list results from recursion.
