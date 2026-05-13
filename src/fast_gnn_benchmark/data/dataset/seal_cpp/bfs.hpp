#pragma once

#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "csr.hpp"


// k-hop BFS from a set of seed nodes on the full graph.
// Returns:
//   sub_nodes     : global node indices in discovery order
//   global_to_local : map global_id → index in sub_nodes
std::pair<std::vector<int>, std::unordered_map<int, int>>
bfs_k_hop(const CSRGraph& csr, const std::vector<int>& seeds, int k) {
    std::vector<int> sub_nodes;
    std::unordered_map<int, int> g2l;
    std::unordered_set<int> visited;
    std::deque<std::pair<int, int>> queue;  // (node, depth)

    for (int s : seeds) {
        if (!visited.count(s)) {
            visited.insert(s);
            g2l[s] = static_cast<int>(sub_nodes.size());
            sub_nodes.push_back(s);
            queue.push_back({s, 0});
        }
    }

    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();

        // Mark the node but do not explore beyond depth k
        if (depth >= k) continue;

        for (int i = csr.row_ptr[node]; i < csr.row_ptr[node + 1]; ++i) {
            int nb = csr.col_idx[i];
            if (!visited.count(nb)) {
                visited.insert(nb);
                g2l[nb] = static_cast<int>(sub_nodes.size());
                sub_nodes.push_back(nb);
                queue.push_back({nb, depth + 1});
            }
        }
    }

    return {sub_nodes, g2l};
}


// BFS distances from source_local on the local subgraph.
// excluded_local is treated as non-existent: never visited, never used as relay.
// Unreachable nodes (including excluded_local itself) receive INT_MAX.
std::vector<int> bfs_distances(
    const std::vector<std::vector<int>>& local_adj,
    int source_local,
    int excluded_local,
    int n
) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<int> dist(n, INF);
    dist[source_local] = 0;

    std::deque<int> queue;
    queue.push_back(source_local);

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();

        for (int nb : local_adj[node]) {
            if (nb == excluded_local) continue;
            if (dist[nb] == INF) {
                dist[nb] = dist[node] + 1;
                queue.push_back(nb);
            }
        }
    }

    return dist;
}
