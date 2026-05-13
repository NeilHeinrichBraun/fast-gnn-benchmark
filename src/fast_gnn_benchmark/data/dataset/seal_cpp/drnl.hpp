#pragma once

#include <algorithm>
#include <climits>
#include <vector>

#include <torch/extension.h>


// DRNL node labeling (Zhang & Chen, 2018).
// z[i] = 1 + min(d1,d2) + floor(d/2) * (floor(d/2) + d%2 - 1)
// where d1 = dist2src[i], d2 = dist2dst[i], d = d1 + d2.
// Nodes unreachable from src or dst (distance == INT_MAX) receive z = 0.
// src_local and dst_local are unconditionally forced to 1.
at::Tensor compute_drnl(
    const std::vector<int>& dist2src,
    const std::vector<int>& dist2dst,
    int src_local,
    int dst_local
) {
    const int n   = static_cast<int>(dist2src.size());
    const int INF = std::numeric_limits<int>::max();

    std::vector<int64_t> z(n, 0);

    for (int i = 0; i < n; ++i) {
        // Guard before computing d = d1 + d2 to avoid overflow.
        if (dist2src[i] == INF || dist2dst[i] == INF) continue;

        const int d     = dist2src[i] + dist2dst[i];
        const int d2    = d / 2;
        const int d_mod = d % 2;
        z[i] = 1 + std::min(dist2src[i], dist2dst[i]) + d2 * (d2 + d_mod - 1);
    }

    // Source and destination are always labelled 1 regardless of distances.
    z[src_local] = 1;
    z[dst_local] = 1;

    return torch::tensor(z, torch::kLong);
}
