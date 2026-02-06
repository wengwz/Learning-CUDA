#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#include <float.h>

template <typename T>
__global__ void trace_kernel(const T* input, T* result, size_t n, size_t cols) {
  size_t step = gridDim.x * blockDim.x;
  size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = start; i < n; i += step) {
    size_t offset = i * cols + i;
    atomicAdd(result, input[offset]);
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function

  T h_result = 0;
  T* d_input, *d_result;

  // allocate device memory
  cudaMalloc(&d_input, rows * cols * sizeof(T));
  cudaMalloc(&d_result, sizeof(T));

  // copy data to device
  cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, &h_result, sizeof(T), cudaMemcpyHostToDevice);
  
  int block_size = 256;
  int grid_size = 256;
  // launch kernel function
  trace_kernel<T><<<dim3(grid_size), dim3(block_size)>>>(d_input, d_result, std::min(rows, cols), cols);

  // copy result from device to host
  cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_result);

  return h_result;
}

template<typename T> __device__ __forceinline__ float get_inf() {
  if constexpr (std::is_same_v<T, float>) {
    return -FLT_MAX;
  } else if constexpr (std::is_same_v<T, __half>) {
    return -65504.0f;
  } else {
    return -FLT_MAX;
  }
};

// CUDA Kernel
template <typename T>
__global__ void attention_kernel(const T* q, const T* k, const T* v, T* o,
                                 int target_seq_len, int src_seq_len, 
                                 int query_heads, int kv_heads, int head_dim, 
                                 float scale, bool is_causal) {
    int b = blockIdx.x; 
    int h = blockIdx.y; 
    int i = blockIdx.z; 

    if (b >= gridDim.x || h >= query_heads || i >= target_seq_len) return;

    int groups = query_heads / kv_heads;
    int kv_h = h / groups;

    extern __shared__ float scores[]; 

    // calculate QK^T
    float max_score = get_inf<T>();
    for (int j = 0; j < src_seq_len; ++j) {
        if (is_causal && i < j) {
            scores[j] = get_inf<T>();
        } else {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float q_val = (float)q[((b * target_seq_len + i) * query_heads + h) * head_dim + d];
                float k_val = (float)k[((b * src_seq_len + j) * kv_heads + kv_h) * head_dim + d];
                sum += q_val * k_val;
            }
            scores[j] = sum * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
    }

    // softmax
    float exp_sum = 0.0f;
    for (int j = 0; j < src_seq_len; ++j) {
        if (scores[j] > get_inf<T>()) {
            scores[j] = expf(scores[j] - max_score);
            exp_sum += scores[j];
        } else {
            scores[j] = 0.0f;
        }
    }

    // compute output O
    for (int d = 0; d < head_dim; ++d) {
        float out_val = 0.0f;
        for (int j = 0; j < src_seq_len; ++j) {
            float v_val = (float)v[((b * src_seq_len + j) * kv_heads + kv_h) * head_dim + d];
            out_val += (scores[j] / exp_sum) * v_val;
        }

        o[((b * target_seq_len + i) * query_heads + h) * head_dim + d] = (T)out_val;
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);

    // allocate device memory
    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, k_size);
    cudaMalloc(&d_v, v_size);
    cudaMalloc(&d_o, o_size);

    // copy data to device
    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);

    float scale_fac = 1.0 / std::sqrt(head_dim);
    
    dim3 gridDim(batch_size, query_heads, target_seq_len);
    dim3 blockDim(1); 
    size_t smem_size = src_seq_len * sizeof(float);

    // launch kernel function
    attention_kernel<<<gridDim, blockDim, smem_size>>>(
        d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, 
        query_heads, kv_heads, head_dim, scale_fac, is_causal
    );

    // copy result from device to host
    cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q); 
    cudaFree(d_k); 
    cudaFree(d_v); 
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
