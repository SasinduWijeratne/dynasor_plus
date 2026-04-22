// ============================================================================
//  dynasor_ooc.h -- Out-of-core (streaming) MTTKRP and CP-ALS support.
//
//  When the sparse tensor is larger than available RAM, none of the in-core
//  layouts can be populated.  We keep the tensor on disk as a .dnb cache and
//  stream fixed-size chunks of nonzeros through RAM once per CP-ALS iteration.
//  The all-modes single-pass MTTKRP kernel reads the tensor once per iter and
//  updates every Yhat[n], minimizing disk traffic.
//
//  Data layout on disk matches dynasor_io.cpp (.dnb): header, vals[nnz],
//  idx[0][nnz]..idx[N-1][nnz].  Each chunk needs N+1 seek+fread pairs.
// ============================================================================
#ifndef DYNASOR_OOC_H
#define DYNASOR_OOC_H

#include "dynasor_common.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace dynasor {

class OocStream {
public:
    OocStream();
    ~OocStream();

    OocStream(const OocStream&)            = delete;
    OocStream& operator=(const OocStream&) = delete;

    bool open(const Tensor& T,
              const std::string& path,
              uint64_t chunk_nnz);
    void close();

    uint64_t read_chunk(uint64_t b);

    const value_t*         vals()        const { return vals_; }
    const idx_t* const*    idx()         const { return idx_ptrs_; }
    idx_t*                 idx_col(int n) const { return idx_cols_[(size_t)n]; }
    uint64_t               chunk_nnz()   const { return chunk_nnz_; }
    uint64_t               total_nnz()   const { return total_nnz_; }
    int                    num_modes()   const { return num_modes_; }
    bool                   is_open()     const { return fp_ != nullptr; }
    size_t                 buffer_bytes()const { return buf_bytes_; }

private:
    std::FILE*          fp_           = nullptr;
    uint64_t            header_bytes_ = 0;
    uint64_t            chunk_nnz_    = 0;
    uint64_t            total_nnz_    = 0;
    int                 num_modes_    = 0;

    void*               buf_raw_      = nullptr;
    size_t              buf_bytes_    = 0;
    value_t*            vals_         = nullptr;
    idx_t*              idx_cols_[DYN_MAX_MODES] = {nullptr};
    const idx_t*        idx_ptrs_[DYN_MAX_MODES]   = {nullptr};
};

bool ooc_load_header(const std::string& path, Tensor& T);
bool ooc_precompute_norm(Tensor& T);
uint64_t ooc_default_chunk_nnz(const Tensor& T);

} // namespace dynasor

#endif // DYNASOR_OOC_H
