#include <torch/extension.h>

#include <tuple>
#include <unordered_map>
#include <vector>

#include "bfs.hpp"
#include "csr.hpp"
#include "drnl.hpp"


// ---------------------------------------------------------------------------
// CSR construction
// ---------------------------------------------------------------------------

// Builds a CSR representation from a COO edge_index tensor.
// Returns (row_ptr [num_nodes+1, int32], col_idx [num_edges, int32]).
// Stored as int32 to halve memory; node/edge counts in OGBL fit comfortably.
std::tuple<at::Tensor, at::Tensor>
build_csr_from_edge_index(const at::Tensor& edge_index, int64_t num_nodes) {
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
                "edge_index must have shape [2, E]");

    auto ei       = edge_index.contiguous().to(torch::kCPU).to(torch::kLong);
    const int64_t E = ei.size(1);
    const auto*  src_data = ei.data_ptr<int64_t>();        // row 0
    const auto*  dst_data = src_data + E;                  // row 1

    // --- degree count ---
    std::vector<int> degree(num_nodes, 0);
    for (int64_t i = 0; i < E; ++i)
        degree[static_cast<int>(src_data[i])]++;

    // --- prefix sum → row_ptr ---
    auto row_ptr_t = torch::zeros({num_nodes + 1}, torch::kInt32);
    auto* rp       = row_ptr_t.data_ptr<int>();
    for (int64_t u = 0; u < num_nodes; ++u)
        rp[u + 1] = rp[u] + degree[u];

    // --- fill col_idx ---
    auto col_idx_t = torch::empty({E}, torch::kInt32);
    auto* ci       = col_idx_t.data_ptr<int>();
    std::vector<int> cursor(rp, rp + num_nodes);   // per-row write cursors
    for (int64_t i = 0; i < E; ++i) {
        int src = static_cast<int>(src_data[i]);
        int dst = static_cast<int>(dst_data[i]);
        ci[cursor[src]++] = dst;
    }

    return {row_ptr_t, col_idx_t};
}


// ---------------------------------------------------------------------------
// Local subgraph construction (§3.3)
// ---------------------------------------------------------------------------

// Builds the local adjacency list and sub_edge_index for one (src, dst) pair.
// The target link (src_local ↔ dst_local) is removed from both outputs.
static std::pair<std::vector<std::vector<int>>, at::Tensor>
build_local_adj(
    const std::vector<int>&              sub_nodes,
    const std::unordered_map<int, int>&  g2l,
    const CSRGraph&                      csr,
    int                                  src_local,
    int                                  dst_local
) {
    const int n = static_cast<int>(sub_nodes.size());
    std::vector<std::vector<int>> local_adj(n);
    std::vector<int64_t> rows, cols;

    for (int u_local = 0; u_local < n; ++u_local) {
        int u = sub_nodes[u_local];
        for (int i = csr.row_ptr[u]; i < csr.row_ptr[u + 1]; ++i) {
            int v = csr.col_idx[i];
            auto it = g2l.find(v);
            if (it == g2l.end()) continue;      // v not in subgraph
            int v_local = it->second;

            // Remove target link in both directions
            if ((u_local == src_local && v_local == dst_local) ||
                (u_local == dst_local && v_local == src_local)) continue;

            local_adj[u_local].push_back(v_local);
            rows.push_back(u_local);
            cols.push_back(v_local);
        }
    }

    at::Tensor sub_edge_index;
    if (rows.empty()) {
        sub_edge_index = torch::zeros({2, 0}, torch::kLong);
    } else {
        const int64_t E_local = static_cast<int64_t>(rows.size());
        sub_edge_index = torch::empty({2, E_local}, torch::kLong);
        auto* r = sub_edge_index.data_ptr<int64_t>();   // row 0
        auto* c = r + E_local;                          // row 1
        for (int64_t i = 0; i < E_local; ++i) {
            r[i] = rows[i];
            c[i] = cols[i];
        }
    }

    return {std::move(local_adj), sub_edge_index};
}


// ---------------------------------------------------------------------------
// Batch extraction (§3.5)
// ---------------------------------------------------------------------------

std::tuple<
    std::vector<at::Tensor>,   // sub_edge_indices  [2, E_i] per pair
    std::vector<at::Tensor>,   // z_labels          [n_i]    per pair
    std::vector<at::Tensor>    // sub_nodes         [n_i]    per pair (global indices)
>
batch_extract(
    const at::Tensor& row_ptr_t,
    const at::Tensor& col_idx_t,
    const at::Tensor& src_batch,
    const at::Tensor& dst_batch,
    int64_t           num_hops,
    int64_t           num_nodes
) {
    TORCH_CHECK(row_ptr_t.is_contiguous() && row_ptr_t.scalar_type() == torch::kInt32,
                "row_ptr must be a contiguous int32 tensor");
    TORCH_CHECK(col_idx_t.is_contiguous() && col_idx_t.scalar_type() == torch::kInt32,
                "col_idx must be a contiguous int32 tensor");
    TORCH_CHECK(src_batch.size(0) == dst_batch.size(0),
                "src_batch and dst_batch must have the same length");

    const int N = static_cast<int>(src_batch.size(0));

    auto src_cpu = src_batch.contiguous().to(torch::kCPU).to(torch::kLong);
    auto dst_cpu = dst_batch.contiguous().to(torch::kCPU).to(torch::kLong);
    const auto* src_ptr = src_cpu.data_ptr<int64_t>();
    const auto* dst_ptr = dst_cpu.data_ptr<int64_t>();

    // Non-owning CSR view over the Python-side tensors
    CSRGraph csr{
        row_ptr_t.data_ptr<int>(),
        col_idx_t.data_ptr<int>(),
        static_cast<int>(num_nodes),
        static_cast<int>(col_idx_t.numel())
    };

    std::vector<at::Tensor> sub_edges(N), z_labels(N), sub_nodes_out(N);

    for (int i = 0; i < N; ++i) {
        const int src = static_cast<int>(src_ptr[i]);
        const int dst = static_cast<int>(dst_ptr[i]);

        auto [sub_nodes, g2l] = bfs_k_hop(csr, {src, dst}, static_cast<int>(num_hops));

        const int src_local = g2l.at(src);
        const int dst_local = g2l.at(dst);
        const int n         = static_cast<int>(sub_nodes.size());

        auto [local_adj, sub_edge_index] =
            build_local_adj(sub_nodes, g2l, csr, src_local, dst_local);

        auto dist2src = bfs_distances(local_adj, src_local, dst_local, n);
        auto dist2dst = bfs_distances(local_adj, dst_local, src_local, n);

        sub_edges[i]     = sub_edge_index;
        z_labels[i]      = compute_drnl(dist2src, dist2dst, src_local, dst_local);
        sub_nodes_out[i] = torch::tensor(
            std::vector<int64_t>(sub_nodes.begin(), sub_nodes.end()),
            torch::kLong
        );
    }

    return {sub_edges, z_labels, sub_nodes_out};
}


// ---------------------------------------------------------------------------
// pybind11 registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_csr", &build_csr_from_edge_index,
          "Build CSR (row_ptr, col_idx) from edge_index tensor",
          py::arg("edge_index"), py::arg("num_nodes"));
    m.def("batch_extract", &batch_extract,
          "SEAL batch subgraph extraction + DRNL labeling",
          py::arg("row_ptr"), py::arg("col_idx"),
          py::arg("src_batch"), py::arg("dst_batch"),
          py::arg("num_hops"), py::arg("num_nodes"));
}
