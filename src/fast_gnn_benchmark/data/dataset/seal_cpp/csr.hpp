#pragma once

// Non-owning view over CSR arrays (row_ptr, col_idx) held by the caller.
// Build the actual int arrays in seal_ops.cpp and pass raw pointers here.
struct CSRGraph {
    const int* row_ptr;  // [num_nodes + 1]
    const int* col_idx;  // [num_edges]
    int num_nodes;
    int num_edges;
};
