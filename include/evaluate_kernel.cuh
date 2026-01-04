/**
 * @file evaluate_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of device functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CUDA_EVALUATE_CUH
#define CUDA_EVALUATE_CUH

#include <mma.h>

#include "patch_base.cuh"

using namespace nvcuda;

// #define PRINT_INFO
// #define TIMING

namespace PSMF
{

  template <int dim_m, int dim_n = dim_m, int dim_z = -1>
  struct Shape
  {
    static constexpr int m = dim_m;
    static constexpr int n = dim_n;
    static constexpr int z = dim_z;
  };

  ////////////////////////////////////////////////////////////////////
  /////////////////////// TPEvaluatorBase ////////////////////////////
  ////////////////////////////////////////////////////////////////////
  /**
   * A base struct for the various TensorProduct Evaluator template
   * specializations, containing common functionalities.
   *
   * @tparam T Type of the actual vectorized array. We are using the
   *   Couriously Recurring Template Pattern (see
   *   https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) in
   *   this struct to avoid having to resort to `virtual` member functions.
   */
  template <typename T,
            int fe_degree,
            typename Number,
            LaplaceVariant laplace_type,
            int            dim>
  struct TPEvaluatorBase
  {
    __device__
    TPEvaluatorBase() = default;

    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {}

    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {}

    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *mass_matrix,
            const Number *laplace_matrix,
            Number       *tmp)
    {}

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {}

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {}
  };

  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::Basic, 2>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid_y = threadIdx.y % n_dofs_1d;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_dofs_1d + tid_x;

      const int n_active_t =
        direction == 0 ? shapeA::m * shapeB::m : shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeB::m : tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeB::m : tid % shapeB::n;

      constexpr int reduction = direction == 0 ? shapeA::n : shapeA::n;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < reduction; ++k)
        {
          const unsigned int shape_idx =
            transposed ? k * reduction + row : row * reduction + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * shapeB::n + k) : (k * shapeB::n + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * shapeA::m + row) : (row * shapeB::n + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid_y = threadIdx.y % n_dofs_1d;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_dofs_1d + tid_x;

      const int n_active_t =
        direction == 0 ? shapeA::m * shapeB::m : shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeB::m : tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeB::m : tid % shapeB::n;

      constexpr int reduction = direction == 0 ? shapeA::n : shapeA::n;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < reduction; ++k)
        {
          const unsigned int shape_idx = row * reduction + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * shapeB::n + k) : (k * shapeB::n + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * shapeA::m + row) : (row * shapeB::n + col);

      if (add)
        {
          if (atomicop)
            atomicAdd(&out[destination_idx], pval);
          else
            out[destination_idx] += pval;
        }
      else if (sub)
        {
          if (atomicop)
            atomicAdd(&out[destination_idx], -pval);
          else
            out[destination_idx] -= pval;
        }
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::Basic, 3>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid_y = threadIdx.y % n_dofs_1d;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_dofs_1d + tid_x;

#ifdef CONFLICTFREE
      const int n_active_t = direction == 0 ? shapeA::m * shapeB::m :
                             direction == 1 ? shapeA::m * shapeB::n :
                                              shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeA::m :
                      direction == 1 ? tid / shapeB::n :
                                       tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeA::m :
                      direction == 1 ? tid % shapeB::n :
                                       tid % shapeB::n;

      constexpr int stride  = direction == 0 ? shapeB::m * shapeB::n :
                              direction == 1 ? shapeB::m * shapeB::n :
                                               shapeB::m * shapeB::n;
      constexpr int stride1 = direction == 0 ? shapeA::m * shapeB::m :
                              direction == 1 ? shapeA::m * shapeB::n :
                                               shapeA::m * shapeB::n;
      constexpr int dim_z   = direction == 0 ? shapeB::z :
                              direction == 1 ? shapeB::z :
                                               shapeB::z;

      constexpr int reduction = transposed ? shapeA::m : shapeA::n;

      Number pval[dim_z];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx = transposed ?
                                      ((direction == 0) ? k * reduction + col :
                                       (direction == 1) ? k * reduction + row :
                                                          k * reduction + z) :
                                      ((direction == 0) ? col * reduction + k :
                                       (direction == 1) ? row * reduction + k :
                                                          z * reduction + k);

              const int source_idx =
                (direction == 0) ? (row * shapeB::n + k + z * stride) :
                (direction == 1) ? (k * shapeB::n + col + z * stride) :
                                   (row * shapeB::n + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < dim_z; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (row * shapeA::m + col + z * stride1) :
            (direction == 1) ? (row * shapeB::n + col + z * stride1) :
                               (row * shapeB::n + col + z * stride1);

          if constexpr (add)
            out[destination_idx] += pval[z];
          else if constexpr (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
#else
      const int n_active_t = direction == 0 ? shapeA::m * shapeB::m :
                             direction == 1 ? shapeA::m * shapeB::n :
                                              shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeB::m :
                      direction == 1 ? tid / shapeB::n :
                                       tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeB::m :
                      direction == 1 ? tid % shapeB::n :
                                       tid % shapeB::n;

      constexpr int stride = direction == 0 ? shapeB::m * shapeB::n :
                             direction == 1 ? shapeB::m * shapeB::n :
                                              shapeB::m * shapeB::n;
      constexpr int dim_z  = direction == 0 ? shapeB::z :
                             direction == 1 ? shapeB::z :
                                              shapeB::z;

      constexpr int reduction = transposed ? shapeA::m : shapeA::n;

      Number pval[dim_z];
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx =
                transposed ? k * reduction + row : row * reduction + k;

              const int source_idx =
                (direction == 0) ? (col * shapeB::n + k + z * stride) :
                (direction == 1) ? (k * shapeB::n + col + z * stride) :
                                   (z * shapeB::n + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < dim_z; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (col * shapeB::n + row + z * stride) :
            (direction == 1) ? (row * shapeB::n + col + z * stride) :
                               (z * shapeB::n + col + row * stride);

          if constexpr (add)
            out[destination_idx] += pval[z];
          else if constexpr (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
#endif
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool transposed,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid_y = threadIdx.y % n_dofs_1d;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_dofs_1d + tid_x;

#ifdef CONFLICTFREE
      const int n_active_t = direction == 0 ? shapeA::m * shapeB::m :
                             direction == 1 ? shapeA::m * shapeB::n :
                                              shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeA::m :
                      direction == 1 ? tid / shapeB::n :
                                       tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeA::m :
                      direction == 1 ? tid % shapeB::n :
                                       tid % shapeB::n;

      constexpr int stride  = direction == 0 ? shapeB::m * shapeB::n :
                              direction == 1 ? shapeB::m * shapeB::n :
                                               shapeB::m * shapeB::n;
      constexpr int stride1 = direction == 0 ? shapeA::m * shapeB::m :
                              direction == 1 ? shapeA::m * shapeB::n :
                                               shapeA::m * shapeB::n;
      constexpr int dim_z   = direction == 0 ? shapeB::z :
                              direction == 1 ? shapeB::z :
                                               shapeB::z;

      constexpr int reduction = transposed ? shapeA::m : shapeA::n;

      Number pval[dim_z];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < shapeA::n; ++k)
            {
              const int shape_idx = transposed ?
                                      ((direction == 0) ? k * reduction + col :
                                       (direction == 1) ? k * reduction + row :
                                                          k * reduction + z) :
                                      ((direction == 0) ? col * reduction + k :
                                       (direction == 1) ? row * reduction + k :
                                                          z * reduction + k);

              const int source_idx =
                (direction == 0) ? (row * shapeB::n + k + z * stride) :
                (direction == 1) ? (k * shapeB::n + col + z * stride) :
                                   (row * shapeB::n + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < dim_z; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (row * shapeA::m + col + z * stride1) :
            (direction == 1) ? (row * shapeB::n + col + z * stride1) :
                               (row * shapeB::n + col + z * stride1);

          if constexpr (add)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], pval[z]);
              else
                out[destination_idx] += pval[z];
            }
          else if constexpr (sub)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], -pval[z]);
              else
                out[destination_idx] -= pval[z];
            }
          else
            out[destination_idx] = pval[z];
        }
#else
      const int n_active_t = direction == 0 ? shapeA::m * shapeB::m :
                             direction == 1 ? shapeA::m * shapeB::n :
                                              shapeA::m * shapeB::n;

      if (tid >= n_active_t)
        return;

      const int row = direction == 0 ? tid / shapeB::m :
                      direction == 1 ? tid / shapeB::n :
                                       tid / shapeB::n;
      const int col = direction == 0 ? tid % shapeB::m :
                      direction == 1 ? tid % shapeB::n :
                                       tid % shapeB::n;

      constexpr int stride  = direction == 0 ? shapeB::m * shapeB::n :
                              direction == 1 ? shapeB::m * shapeB::n :
                                               shapeB::m * shapeB::n;
      constexpr int stride1 = direction == 0 ? shapeA::m * shapeB::m :
                              direction == 1 ? shapeA::m * shapeB::n :
                                               shapeA::m * shapeB::n;
      constexpr int dim_z   = direction == 0 ? shapeB::z :
                              direction == 1 ? shapeB::z :
                                               shapeB::z;

      constexpr int reduction = shapeA::n;

      Number pval[dim_z];
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx =
                transposed ? k * shapeA::m + row : row * reduction + k;

              const int source_idx =
                (direction == 0) ? (col * shapeB::n + k + z * stride) :
                (direction == 1) ? (k * shapeB::n + col + z * stride) :
                                   (z * shapeB::n + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < dim_z; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (col * shapeA::m + row + z * stride1) :
            (direction == 1) ? (row * shapeB::n + col + z * stride1) :
                               (z * shapeB::n + col + row * stride1);

          if constexpr (add)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], pval[z]);
              else
                out[destination_idx] += pval[z];
            }
          else if constexpr (sub)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], -pval[z]);
              else
                out[destination_idx] -= pval[z];
            }
          else
            out[destination_idx] = pval[z];
        }
#endif
    }
  };


  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::BasicPadding, 2>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx =
            transposed ? k * n_dofs_1d + row : row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx = row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };


  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::BasicPadding, 3>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      constexpr int dim_z     = direction == 2 ? n_dofs_1d : n_dofs_1d - 1;
      constexpr int stride    = n_dofs_1d * n_dofs_1d;
      constexpr int reduction = direction == 2 ? n_dofs_1d - 1 : n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx =
                transposed ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (direction < 2)
        for (int z = 0; z < dim_z; ++z)
          {
            const int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);

            if constexpr (add)
              out[destination_idx] += pval[z];
            else if constexpr (sub)
              out[destination_idx] -= pval[z];
            else
              out[destination_idx] = pval[z];
          }
      else if (row < n_dofs_1d - 1)
        for (int z = 0; z < dim_z; ++z)
          {
            const int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);

            if constexpr (add)
              out[destination_idx] += pval[z];
            else if constexpr (sub)
              out[destination_idx] -= pval[z];
            else
              out[destination_idx] = pval[z];
          }
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool transposed,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      constexpr int dim_z     = direction == 2 ? n_dofs_1d : n_dofs_1d - 1;
      constexpr int stride    = n_dofs_1d * n_dofs_1d;
      constexpr int reduction = direction == 2 ? n_dofs_1d - 1 : n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx =
                transposed ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      __syncthreads();

      if (direction < 2)
        for (int z = 0; z < n_dofs_1d; ++z)
          {
            const int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);

            if constexpr (add)
              {
                if constexpr (atomicop)
                  atomicAdd(&out[destination_idx], pval[z]);
                else
                  out[destination_idx] += pval[z];
              }
            else if constexpr (sub)
              {
                if constexpr (atomicop)
                  atomicAdd(&out[destination_idx], -pval[z]);
                else
                  out[destination_idx] -= pval[z];
              }
            else
              out[destination_idx] = pval[z];
          }
      else if (row < n_dofs_1d - 1)
        for (int z = 0; z < n_dofs_1d; ++z)
          {
            const int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);

            if constexpr (add)
              {
                if constexpr (atomicop)
                  atomicAdd(&out[destination_idx], pval[z]);
                else
                  out[destination_idx] += pval[z];
              }
            else if constexpr (sub)
              {
                if constexpr (atomicop)
                  atomicAdd(&out[destination_idx], -pval[z]);
                else
                  out[destination_idx] -= pval[z];
              }
            else
              out[destination_idx] = pval[z];
          }
    }
  };



  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::ConflictFree, 2>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx =
            transposed ? k * n_dofs_1d + row : row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (unsigned int k = 0; k < n_dofs_1d; ++k)
        {
          const unsigned int shape_idx = row * n_dofs_1d + k;

          const unsigned int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const unsigned int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };


  template <typename T, int fe_degree, typename Number>
  struct TPEvaluatorBase<T, fe_degree, Number, LaplaceVariant::ConflictFree, 3>
  {
    static constexpr int n_dofs_1d = 2 * fe_degree + 3;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    template <bool sub = false>
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *laplace_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->template vmult_impl<sub>(
        dst, src, mass_matrix, laplace_matrix, tmp);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add       = false,
              bool smoothing = false>
    __device__ void
    vmult_mixed(Number       *dst,
                const Number *src,
                const Number *mass_matrix,
                const Number *derivate_matrix,
                Number       *tmp)
    {
      static_cast<T *>(this)
        ->template vmult_mixed_impl<shapeD,
                                    shapeV,
                                    transposed,
                                    atomicop,
                                    add,
                                    smoothing>(
          dst, src, mass_matrix, derivate_matrix, tmp);
    }

    template <bool sub = false>
    __device__ void
    inverse(Number       *dst,
            Number       *src,
            const Number *eigenvalues,
            const Number *eigenvectors,
            Number       *tmp)
    {
      static_cast<T *>(this)->template inverse_impl<sub>(
        dst, src, eigenvalues, eigenvectors, tmp);
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool add,
              bool sub        = false,
              bool transposed = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      constexpr int dim_z     = direction == 2 ? n_dofs_1d : n_dofs_1d - 1;
      constexpr int stride    = n_dofs_1d * n_dofs_1d;
      constexpr int reduction = direction == 2 ? n_dofs_1d - 1 : n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx = transposed ?
                                      ((direction == 0) ? k * n_dofs_1d + col :
                                       (direction == 1) ? k * n_dofs_1d + row :
                                                          k * n_dofs_1d + z) :
                                      ((direction == 0) ? col * n_dofs_1d + k :
                                       (direction == 1) ? row * n_dofs_1d + k :
                                                          z * n_dofs_1d + k);

              const int source_idx =
                (direction == 0) ? (row * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (row * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < n_dofs_1d - 1; ++z)
        {
          const int destination_idx = row * n_dofs_1d + col + z * stride;

          if constexpr (add)
            out[destination_idx] += pval[z];
          else if constexpr (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }

    template <int direction,
              typename shapeA,
              typename shapeB,
              bool transposed,
              bool atomicop,
              bool add,
              bool sub = false>
    __device__ void
    apply_mixed(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y % n_dofs_1d;
      const int col = threadIdx.x;

      constexpr int dim_z     = direction == 2 ? n_dofs_1d : n_dofs_1d - 1;
      constexpr int stride    = n_dofs_1d * n_dofs_1d;
      constexpr int reduction = direction == 2 ? n_dofs_1d - 1 : n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int z = 0; z < dim_z; ++z)
        {
          pval[z] = 0;
          for (int k = 0; k < reduction; ++k)
            {
              const int shape_idx = transposed ?
                                      ((direction == 0) ? k * n_dofs_1d + col :
                                       (direction == 1) ? k * n_dofs_1d + row :
                                                          k * n_dofs_1d + z) :
                                      ((direction == 0) ? col * n_dofs_1d + k :
                                       (direction == 1) ? row * n_dofs_1d + k :
                                                          z * n_dofs_1d + k);

              const int source_idx =
                (direction == 0) ? (row * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (row * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      __syncthreads();

      for (int z = 0; z < n_dofs_1d - 1; ++z)
        {
          const int destination_idx = row * n_dofs_1d + col + z * stride;

          if constexpr (add)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], pval[z]);
              else
                out[destination_idx] += pval[z];
            }
          else if constexpr (sub)
            {
              if constexpr (atomicop)
                atomicAdd(&out[destination_idx], -pval[z]);
              else
                out[destination_idx] -= pval[z];
            }
          else
            out[destination_idx] = pval[z];
        }
    }
  };



  ////////////////////////////////////////////////////////////////////
  /////////////////// TPEvaluatorStokes ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <LaplaceVariant laplace_type,
            typename Number,
            int fe_degree,
            int dim>
  struct TPEvaluatorStokes
    : public TPEvaluatorBase<
        TPEvaluatorStokes<laplace_type, Number, fe_degree, dim>,
        fe_degree,
        Number,
        laplace_type,
        dim>
  {
    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               Number       *tmp)
    {}

    __device__ void
    vmult_mixed_impl(Number       *dst,
                     const Number *src,
                     const Number *mass_matrix,
                     const Number *derivate_matrix,
                     Number       *tmp)
    {}
  };

  template <LaplaceVariant laplace_type, typename Number, int fe_degree>
  struct TPEvaluatorStokes<laplace_type, Number, fe_degree, 2>
    : public TPEvaluatorBase<
        TPEvaluatorStokes<laplace_type, Number, fe_degree, 2>,
        fe_degree,
        Number,
        laplace_type,
        2>
  {
    static constexpr int n_normal  = 2 * fe_degree + 3;
    static constexpr int n_tangent = 2 * fe_degree + 2;

    template <bool sub>
    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               Number       *tmp)
    {
      using shape0 = Shape<n_normal, n_normal>;
      using shape1 = Shape<n_tangent, n_tangent>;
      using shapev = Shape<n_tangent, n_normal>;

      constexpr int offset = n_normal * n_normal;

      this->template apply<0, shape0, shapev, false>(mass_matrix, src, tmp);
      __syncthreads();
      this->template apply<1, shape1, shapev, false>(&laplace_matrix[offset],
                                                     tmp,
                                                     dst);
      __syncthreads();

      this->template apply<0, shape0, shapev, false>(laplace_matrix, src, tmp);
      __syncthreads();
      this->template apply<1, shape1, shapev, true>(&mass_matrix[offset],
                                                    tmp,
                                                    dst);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add,
              bool smoothing>
    __device__ void
    vmult_mixed_impl(Number       *dst,
                     const Number *src,
                     const Number *mass_matrix,
                     const Number *derivate_matrix,
                     Number       *tmp)
    {
      using shapeM  = Shape<n_tangent, n_tangent>;
      using shapeN  = Shape<shapeV::n, shapeD::m>;
      using shapeNt = Shape<shapeD::n, shapeV::m>;

      if constexpr (smoothing)
        {
          if (transposed)
            this->template apply_mixed<1, shapeV, shapeD, atomicop, add>(
              src, derivate_matrix, dst);
          else
            this->template apply_mixed<0, shapeD, shapeV, atomicop, add>(
              derivate_matrix, src, dst);

          return;
        }

      if (transposed)
        this->template apply_mixed<1, shapeV, shapeD, atomicop, false>(
          src, derivate_matrix, tmp);
      else
        this->template apply_mixed<0, shapeD, shapeV, atomicop, false>(
          derivate_matrix, src, tmp);
      __syncthreads();

      if (transposed)
        this->template apply_mixed<1, shapeM, shapeNt, atomicop, add>(
          mass_matrix, tmp, dst);
      else
        this->template apply_mixed<1, shapeM, shapeN, atomicop, add>(
          mass_matrix, tmp, dst);
    }

    template <bool sub>
    __device__ void
    inverse_impl(Number       *dst,
                 Number       *src,
                 const Number *eigenvalues,
                 const Number *eigenvectors,
                 Number       *tmp)
    {
      using shape0 = Shape<n_normal - 2, n_normal - 2>;
      using shape1 = Shape<n_tangent, n_tangent>;
      using shapev = Shape<n_tangent, n_normal - 2>;

      constexpr int offset = (2 * fe_degree + 3) * (2 * fe_degree + 3);

      const int tid_y = threadIdx.y % n_normal;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_normal + tid_x;

      const int row = tid / (n_normal - 2);
      const int col = tid % (n_normal - 2);

      const bool is_active = tid < (2 * fe_degree + 2) * (2 * fe_degree + 1);

      this->template apply<0, shape0, shapev, false, false, true>(eigenvectors,
                                                                  src,
                                                                  tmp);
      __syncthreads();
      this->template apply<1, shape1, shapev, false, false, true>(
        &eigenvectors[offset], tmp, src);
      __syncthreads();

      if (is_active)
        src[row * (n_normal - 2) + col] /=
          (eigenvalues[n_normal + row] + eigenvalues[col]);
      __syncthreads();

      this->template apply<0, shape0, shapev, false>(eigenvectors, src, tmp);
      __syncthreads();
      this->template apply<1, shape1, shapev, false, sub>(&eigenvectors[offset],
                                                          tmp,
                                                          dst);
    }
  };

  template <LaplaceVariant laplace_type, typename Number, int fe_degree>
  struct TPEvaluatorStokes<laplace_type, Number, fe_degree, 3>
    : public TPEvaluatorBase<
        TPEvaluatorStokes<laplace_type, Number, fe_degree, 3>,
        fe_degree,
        Number,
        laplace_type,
        3>
  {
    static constexpr int n_normal  = 2 * fe_degree + 3;
    static constexpr int n_tangent = 2 * fe_degree + 2;

    template <bool sub>
    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *laplace_matrix,
               Number       *tmp)
    {
      using shape0 = Shape<n_normal, n_normal>;
      using shape1 = Shape<n_tangent, n_tangent>;
      using shapev = Shape<n_tangent, n_normal, n_tangent>;

      constexpr int offset    = n_normal * n_normal;
      constexpr int local_dim = laplace_type == LaplaceVariant::Basic ?
                                  n_normal * n_tangent * n_tangent :
                                  n_normal * n_normal * n_tangent;

      this->template apply<0, shape0, shapev, false>(mass_matrix,
                                                     src,
                                                     &tmp[local_dim]);
      __syncthreads();
      this->template apply<1, shape1, shapev, false>(&mass_matrix[offset],
                                                     &tmp[local_dim],
                                                     tmp);
      __syncthreads();
      this->template apply<2, shape1, shapev, false, sub>(
        &laplace_matrix[offset * 2], tmp, dst);
      __syncthreads();

      this->template apply<1, shape1, shapev, false>(&laplace_matrix[offset],
                                                     &tmp[local_dim],
                                                     tmp);
      __syncthreads();

      this->template apply<0, shape0, shapev, false>(laplace_matrix,
                                                     src,
                                                     &tmp[local_dim]);
      __syncthreads();
      this->template apply<1, shape1, shapev, true>(&mass_matrix[offset],
                                                    &tmp[local_dim],
                                                    tmp);
      __syncthreads();
      this->template apply<2, shape1, shapev, !sub, sub>(
        &mass_matrix[offset * 2], tmp, dst);
    }

    template <typename shapeD,
              typename shapeV,
              bool transposed,
              bool atomicop,
              bool add,
              bool smoothing>
    __device__ void
    vmult_mixed_impl(Number       *dst,
                     const Number *src,
                     const Number *mass_matrix,
                     const Number *derivate_matrix,
                     Number       *tmp)
    {
      using shapeDt = Shape<shapeD::n, shapeD::m>;
      using shapeM  = Shape<n_tangent, n_tangent>;
      using shapeN  = Shape<shapeV::n, shapeD::m, n_tangent>;
      using shapeNt = Shape<shapeD::n, shapeV::m, n_tangent>;

      // smoothing
      if constexpr (smoothing)
        {
          if constexpr (transposed)
            this->template apply_mixed<0,
                                       shapeDt,
                                       shapeV,
                                       transposed,
                                       atomicop,
                                       add>(derivate_matrix, src, dst);
          else
            this->template apply_mixed<0,
                                       shapeD,
                                       shapeV,
                                       transposed,
                                       atomicop,
                                       add>(derivate_matrix, src, dst);

          return;
        }

      constexpr int offset    = n_normal * n_normal;
      constexpr int local_dim = laplace_type == LaplaceVariant::Basic ?
                                  n_normal * n_tangent * n_tangent :
                                  n_normal * n_normal * n_tangent;

      if constexpr (transposed)
        this->template apply_mixed<0,
                                   shapeDt,
                                   shapeV,
                                   transposed,
                                   atomicop,
                                   false>(derivate_matrix, src, tmp);
      else
        this->template apply_mixed<0,
                                   shapeD,
                                   shapeV,
                                   transposed,
                                   atomicop,
                                   false>(derivate_matrix, src, tmp);
      __syncthreads();

      if constexpr (transposed)
        this->template apply_mixed<1,
                                   shapeM,
                                   shapeNt,
                                   transposed,
                                   atomicop,
                                   false>(mass_matrix, tmp, &tmp[local_dim]);
      else
        this->template apply_mixed<1,
                                   shapeM,
                                   shapeN,
                                   transposed,
                                   atomicop,
                                   false>(mass_matrix, tmp, &tmp[local_dim]);
      __syncthreads();

      if constexpr (transposed)
        this
          ->template apply_mixed<2, shapeM, shapeNt, transposed, atomicop, add>(
            &mass_matrix[offset], &tmp[local_dim], dst);
      else
        this
          ->template apply_mixed<2, shapeM, shapeN, transposed, atomicop, add>(
            &mass_matrix[offset], &tmp[local_dim], dst);
    }

    template <bool sub>
    __device__ void
    inverse_impl(Number       *dst,
                 Number       *src,
                 const Number *eigenvalues,
                 const Number *eigenvectors,
                 Number       *tmp)
    {
      using shape0 = Shape<n_normal - 2, n_normal - 2>;
      using shape1 = Shape<n_tangent, n_tangent>;
      using shapev = Shape<n_tangent, n_normal - 2, n_tangent>;

      constexpr int offset = (2 * fe_degree + 3) * (2 * fe_degree + 3);

      const int tid_y = threadIdx.y % n_normal;
      const int tid_x = threadIdx.x;
      const int tid   = tid_y * n_normal + tid_x;

      const int row = tid / (n_normal - 2);
      const int col = tid % (n_normal - 2);

      const bool is_active = tid < (2 * fe_degree + 2) * (2 * fe_degree + 1);

      this->template apply<0, shape0, shapev, false, false, true>(eigenvectors,
                                                                  src,
                                                                  tmp);
      __syncthreads();
      this->template apply<1, shape1, shapev, false, false, true>(
        &eigenvectors[offset], tmp, src);
      __syncthreads();
      this->template apply<2, shape1, shapev, false, false, true>(
        &eigenvectors[offset * 2], src, tmp);
      __syncthreads();

      if (is_active)
        for (int z = 0; z < n_tangent; ++z)
          tmp[z * (n_normal - 2) * n_tangent + row * (n_normal - 2) + col] /=
            (eigenvalues[n_normal * 2 + z] + eigenvalues[n_normal + row] +
             eigenvalues[col]);
      __syncthreads();

      this->template apply<0, shape0, shapev, false>(eigenvectors, tmp, src);
      __syncthreads();
      this->template apply<1, shape1, shapev, false>(&eigenvectors[offset],
                                                     src,
                                                     tmp);
      __syncthreads();
      this->template apply<2, shape1, shapev, false, sub>(
        &eigenvectors[offset * 2], tmp, dst);
    }
  };


  ////////////////////////////////////////////////////////////////////
  //////////////////// TPEvaluatorSmoother ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  // template <typename Number,
  //           int            n_dofs_1d,
  //           LaplaceVariant laplace_type,
  //           int            dim>
  // struct TPEvaluatorSmootherVmult
  //   : TPEvaluatorBase<
  //       TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
  //       n_dofs_1d,
  //       Number,
  //       laplace_type,
  //       dim>
  // {
  //   using TPEvaluatorBase<
  //     TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
  //     n_dofs_1d,
  //     Number,
  //     laplace_type,
  //     dim>::apply;

  //   __device__ void
  //   vmult_impl(Number       *dst,
  //              const Number *src,
  //              const Number *mass_matrix,
  //              const Number *laplace_matrix,
  //              Number       *tmp)
  //   {}
  // };


  // template <typename Number, int n_dofs_1d, LaplaceVariant laplace_type>
  // struct TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>
  //   : TPEvaluatorBase<
  //       TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
  //       n_dofs_1d,
  //       Number,
  //       laplace_type,
  //       2>
  // {
  //   using TPEvaluatorBase<
  //     TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
  //     n_dofs_1d,
  //     Number,
  //     laplace_type,
  //     2>::apply;

  //   __device__ void
  //   vmult_impl(Number       *dst,
  //              const Number *src,
  //              const Number *mass_matrix,
  //              const Number *laplace_matrix,
  //              const Number *bilaplace_matrix,
  //              Number       *tmp)
  //   {
  //     apply<0, false>(mass_matrix, src, tmp);
  //     __syncthreads();
  //     apply<1, false, true>(bilaplace_matrix, tmp, dst);
  //     __syncthreads();
  //     apply<0, false>(bilaplace_matrix, src, tmp);
  //     __syncthreads();
  //     apply<1, false, true>(mass_matrix, tmp, dst);
  //     __syncthreads();
  //     apply<0, false>(laplace_matrix, src, tmp);
  //     __syncthreads();
  //     apply<1, false, true, true>(laplace_matrix, tmp, dst);
  //   }
  // };


  // template <typename Number,
  //           int                n_dofs_1d,
  //           SmootherVariant    smoother,
  //           LocalSolverVariant local_solver,
  //           int                dim>
  // struct TPEvaluatorSmootherInv
  // {
  //   __device__ void
  //   apply_inverse(Number       *dst,
  //                 Number       *src,
  //                 const Number *eigenvalues,
  //                 const Number *eigenvectors,
  //                 Number       *tmp)
  //   {}

  //   template <int direction, bool contract_over_rows, bool add = false>
  //   __device__ void
  //   apply(const Number *shape_data, const Number *in, Number *out)
  //   {}
  // };


  // // Exact. TODO:
  // template <typename Number, int n_dofs_1d, SmootherVariant smoother>
  // struct TPEvaluatorSmootherInv<Number,
  //                               n_dofs_1d,
  //                               smoother,
  //                               LocalSolverVariant::Direct,
  //                               2>
  // {
  //   __device__ void
  //   apply_inverse(Number       *dst,
  //                 Number       *src,
  //                 const Number *eigenvalues,
  //                 const Number *eigenvectors,
  //                 Number       *tmp)
  //   {}

  //   template <int direction, bool contract_over_rows, bool add = false>
  //   __device__ void
  //   apply(const Number *shape_data, const Number *in, Number *out)
  //   {}
  // };

  // // Bila, KSVD
  // template <typename Number,
  //           int                n_dofs_1d,
  //           SmootherVariant    smoother,
  //           LocalSolverVariant local_solver>
  // struct TPEvaluatorSmootherInv<Number, n_dofs_1d, smoother, local_solver, 2>
  // {
  //   __device__ void
  //   apply_inverse(Number       *dst,
  //                 Number       *src,
  //                 const Number *eigenvalues,
  //                 const Number *eigenvectors,
  //                 Number       *tmp)
  //   {
  //     constexpr unsigned int n_cols =
  //       smoother == SmootherVariant::GLOBAL ? n_dofs_1d : n_dofs_1d - 2;
  //     constexpr unsigned int offset = n_cols * n_cols;

  //     const unsigned int linear_tid =
  //       threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;
  //     const unsigned int row = linear_tid / n_cols;
  //     const unsigned int col = linear_tid % n_cols;

  //     const bool is_active = linear_tid < Util::pow(n_cols, 2);

  //     apply<0, true>(eigenvectors, src, tmp);
  //     __syncthreads();
  //     apply<1, true>(&eigenvectors[offset], tmp, src);
  //     __syncthreads();
  //     if (is_active)
  //       src[row * n_cols + col] /=
  //         (1 + eigenvalues[n_cols + row] * eigenvalues[col]);
  //     __syncthreads();
  //     apply<0, false>(eigenvectors, src, tmp);
  //     __syncthreads();
  //     apply<1, false, true>(&eigenvectors[offset], tmp, dst);
  //   }

  //   template <int direction, bool contract_over_rows, bool add = false>
  //   __device__ void
  //   apply(const Number *shape_data, const Number *in, Number *out)
  //   {
  //     constexpr unsigned int n_cols =
  //       smoother == SmootherVariant::GLOBAL ? n_dofs_1d : n_dofs_1d - 2;

  //     const unsigned int linear_tid =
  //       threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;
  //     const unsigned int row = linear_tid / n_cols;
  //     const unsigned int col = linear_tid % n_cols;

  //     const bool is_active = linear_tid < Util::pow(n_cols, 2);

  //     Number pval = 0;

  //     // kernel product: A kdot src, [N x N] * [N^dim, 1]
  //     // #pragma unroll
  //     if (is_active)
  //       for (unsigned int k = 0; k < n_cols; ++k)
  //         {
  //           const unsigned int shape_idx =
  //             contract_over_rows ?
  //               ((direction == 0) ? k * n_cols + col : k * n_cols + row) :
  //               ((direction == 0) ? col * n_cols + k : row * n_cols + k);

  //           const unsigned int source_idx =
  //             (direction == 0) ? (row * n_cols + k) : (k * n_cols + col);

  //           pval += shape_data[shape_idx] * in[source_idx];
  //         }

  //     if (is_active)
  //       {
  //         const unsigned int destination_idx = row * n_cols + col;

  //         if (add)
  //           out[destination_idx] += pval;
  //         else
  //           out[destination_idx] = pval;
  //       }
  //   }
  // };

  // template <typename Number, int n_dofs_1d, LocalSolverVariant local_solver>
  // struct TPEvaluatorSmootherInv<Number, n_dofs_1d, local_solver, 3>
  // {
  //   __device__ void
  //   apply_inverse(Number       *dst,
  //                 Number       *src,
  //                 const Number *eigenvalues,
  //                 const Number *eigenvectors,
  //                 Number       *tmp)
  //   {
  //     constexpr unsigned int local_dim = Util::pow(n_dofs_1d, 3);

  //     apply<0, true>(eigenvectors, src, tmp);
  //     __syncthreads();
  //     apply<1, true>(eigenvectors, tmp, &tmp[local_dim]);
  //     __syncthreads();
  //     apply<2, true>(eigenvectors, &tmp[local_dim], tmp);
  //     __syncthreads();
  //     for (unsigned int z = 0; z < n_dofs_1d; ++z)
  //       {
  //         tmp[z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
  //             threadIdx.x % n_dofs_1d] /=
  //           (eigenvalues[z] + eigenvalues[threadIdx.y] +
  //            eigenvalues[threadIdx.x % n_dofs_1d]);
  //       }
  //     __syncthreads();
  //     apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
  //     __syncthreads();
  //     apply<1, false>(eigenvectors, &tmp[local_dim], tmp);
  //     __syncthreads();
  //     apply<2, false, true>(eigenvectors, tmp, dst);
  //   }

  //   template <int direction, bool contract_over_rows, bool add = false>
  //   __device__ void
  //   apply(const Number *shape_data, const Number *in, Number *out)
  //   {
  //     constexpr unsigned int stride = n_dofs_1d * n_dofs_1d;

  //     const unsigned int row = threadIdx.y;
  //     const unsigned int col = threadIdx.x % n_dofs_1d;

  //     Number pval[n_dofs_1d];

  //     // kernel product: A kdot src, [N x N] * [N^dim, 1]
  //     for (unsigned int z = 0; z < n_dofs_1d; ++z)
  //       {
  //         pval[z] = 0;
  //         // #pragma unroll
  //         for (unsigned int k = 0; k < n_dofs_1d; ++k)
  //           {
  //             const unsigned int shape_idx =
  //               contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d +
  //               k;

  //             const unsigned int source_idx =
  //               (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
  //               (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
  //                                  (z * n_dofs_1d + col + k * stride);

  //             pval[z] += shape_data[shape_idx] * in[source_idx];
  //           }
  //       }

  //     for (unsigned int z = 0; z < n_dofs_1d; ++z)
  //       {
  //         const unsigned int destination_idx =
  //           (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
  //           (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
  //                              (z * n_dofs_1d + col + row * stride);
  //         if (add)
  //           out[destination_idx] += pval[z];
  //         else
  //           out[destination_idx] = pval[z];
  //       }
  //   }
  // };


  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __device__ void
  evaluate_laplace(
    const unsigned int                                             local_patch,
    SharedDataOp<dim, Number, laplace>                            *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_2d * dim;

    const int tid_y  = threadIdx.y % n_dofs_1d;
    const int tid_yy = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x  = threadIdx.x;
    const int tid    = tid_y * n_dofs_1d + tid_x;
    const int tid_g  = tid_yy * n_dofs_1d + tid_x;

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    TPEvaluatorStokes<laplace, Number, fe_degree, dim> eval;
    __syncthreads();

    using shapeB = Shape<n_dofs_1d, n_dofs_1d - 1>;
    using shapeU = Shape<n_dofs_1d - 1, n_dofs_1d, n_dofs_1d - 1>;
    using shapeP = Shape<n_dofs_1d - 1, n_dofs_1d - 1, n_dofs_1d - 1>;

    // B^T * U
    eval.template vmult_mixed<shapeB, shapeU, true, false>(
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * dim * (dim - 1) +
                                   component * n_dofs_2d * (dim - 1)],
      &shared_data
         ->local_mix_der[local_patch * n_dofs_2d * dim + component * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim * (dim - 1)]);
    __syncthreads();

#if MEMTYPE == 0
    const unsigned int *mapping = component == 0 ? ltoh_dgn :
                                  component == 1 ? ltoh_dgt :
                                                   ltoh_dgz;
#elif MEMTYPE == 1
    const unsigned int *mapping = component == 0 ? gpu_data->ltoh_dgn :
                                  component == 1 ? gpu_data->ltoh_dgt :
                                                   gpu_data->ltoh_dgz;
#endif

    for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
      if (tid_g + i * block_size < n_patch_dofs_dg)
        shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) + tid_g +
                         i * block_size] = 0;
    __syncthreads();
    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg)
        {
          atomicAdd(&shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                                      tid + i * n_dofs_2d],
                    shared_data->local_dst[local_patch * n_patch_dofs +
                                           component * n_patch_dofs_rt / dim +
                                           mapping[tid + i * n_dofs_2d]]);
        }
    __syncthreads();
    for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
      if (tid_g + i * block_size < n_patch_dofs_dg)
        shared_data->local_dst[local_patch * n_patch_dofs + n_patch_dofs_rt +
                               tid_g + i * block_size] =
          shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) + tid_g +
                           i * block_size];
    __syncthreads();

    // M * U
    eval.vmult(
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_mass[local_patch * n_dofs_2d * dim * dim +
                               component * n_dofs_2d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                  component * n_dofs_2d * dim],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim * (dim - 1)]);
    __syncthreads();

    // B * P
    for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
      if (tid_g + i * block_size < n_patch_dofs_dg)
        {
          shared_data
            ->local_src[local_patch * n_patch_dofs + tid_g + i * block_size] =
            shared_data->local_src[local_patch * n_patch_dofs +
                                   n_patch_dofs_rt + tid_g + i * block_size];
        }
    __syncthreads();

    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg && component != 0)
        {
          shared_data->local_src[local_patch * n_patch_dofs +
                                 component * n_patch_dofs_dg +
                                 mapping[tid + i * n_dofs_2d]] =
            shared_data->local_src[local_patch * n_patch_dofs +
#if MEMTYPE == 0
                                   ltoh_dgn[tid + i * n_dofs_2d]];
#elif MEMTYPE == 1
                                   gpu_data->ltoh_dgn[tid + i * n_dofs_2d]];
#endif
        }
    __syncthreads();

    eval.template vmult_mixed<shapeB, shapeP, false, false, true>(
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data
         ->local_src[local_patch * n_patch_dofs + component * n_patch_dofs_dg],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * dim * (dim - 1) +
                                   component * n_dofs_2d * (dim - 1)],
      &shared_data
         ->local_mix_der[local_patch * n_dofs_2d * dim + component * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim * (dim - 1)]);
    __syncthreads();
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __device__ void
  evaluate_laplace_padding(
    const unsigned int                                             local_patch,
    SharedDataOp<dim, Number, laplace>                            *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_dofs_component =
      Util::pow(n_dofs_1d, dim - 1) * (2 * fe_degree + 2);
    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg  = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs     = n_dofs_component * (dim + 1);
    constexpr int n_patch_dofs_tmp = n_dofs_component * dim;
    constexpr int block_size       = n_dofs_2d * dim;

    const int tid_y  = threadIdx.y % n_dofs_1d;
    const int tid_yy = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x  = threadIdx.x;
    const int tid    = tid_y * n_dofs_1d + tid_x;
    const int tid_g  = tid_yy * n_dofs_1d + tid_x;

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    TPEvaluatorStokes<laplace, Number, fe_degree, dim> eval;
    __syncthreads();

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    //   {
    //     auto shift = n_dofs_2d;
    //     printf("mass0\n");
    //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
    //       {
    //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
    //           printf("%f ", shared_data->local_mix_mass[i * n_dofs_1d + j]);
    //         printf("\n");
    //       }
    //     printf("mass1\n");
    //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
    //       {
    //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
    //           printf("%f ",
    //                  shared_data->local_mix_mass[shift + i * n_dofs_1d + j]);
    //         printf("\n");
    //       }
    //     printf("der0\n");
    //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
    //       {
    //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
    //           printf("%f ", shared_data->local_mix_der[i * n_dofs_1d + j]);
    //         printf("\n");
    //       }
    //     printf("der1\n");
    //     for (unsigned int i = 0; i < n_dofs_1d; ++i)
    //       {
    //         for (unsigned int j = 0; j < n_dofs_1d; ++j)
    //           printf("%f ",
    //                  shared_data->local_mix_der[shift + i * n_dofs_1d + j]);
    //         printf("\n");
    //       }
    //   }


    using shapeB = Shape<n_dofs_1d, n_dofs_1d - 1>;
    using shapeU = Shape<n_dofs_1d - 1, n_dofs_1d, n_dofs_1d - 1>;
    using shapeP = Shape<n_dofs_1d - 1, n_dofs_1d - 1, n_dofs_1d - 1>;

    // B^T * U
    eval.template vmult_mixed<shapeB, shapeU, true, false>(
      &shared_data
         ->local_dst[local_patch * n_patch_dofs + component * n_dofs_component],
      &shared_data
         ->local_src[local_patch * n_patch_dofs + component * n_dofs_component],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * dim * (dim - 1) +
                                   component * n_dofs_2d * (dim - 1)],
      &shared_data
         ->local_mix_der[local_patch * n_dofs_2d * dim + component * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) +
                        component * n_dofs_component * (dim - 1)]);
    __syncthreads();

    const unsigned int *mapping = component == 0 ? gpu_data->ltoh_dgn_p :
                                  component == 1 ? gpu_data->ltoh_dgt_p :
                                                   gpu_data->ltoh_dgz_p;

    for (unsigned int i = 0; i < n_dofs_component / block_size + 1; ++i)
      if (tid_g + i * block_size < n_dofs_component)
        shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) + tid_g +
                         i * block_size] = 0;
    __syncthreads();
    for (unsigned int i = 0; i < n_dofs_component / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_dofs_component)
        {
          atomicAdd(
            &shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) + tid +
                              i * n_dofs_2d],
            shared_data->local_dst[local_patch * n_patch_dofs +
                                   component * n_dofs_component +
                                   mapping[tid + i * n_dofs_2d]]);
        }
    __syncthreads();
    for (unsigned int i = 0; i < n_dofs_component / block_size + 1; ++i)
      if (tid_g + i * block_size < n_dofs_component)
        shared_data
          ->local_dst[local_patch * n_patch_dofs + dim * n_dofs_component +
                      tid_g + i * block_size] =
          shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) + tid_g +
                           i * block_size];
    __syncthreads();

    // M * U
    eval.vmult(&shared_data->local_dst[local_patch * n_patch_dofs +
                                       component * n_dofs_component],
               &shared_data->local_src[local_patch * n_patch_dofs +
                                       component * n_dofs_component],
               &shared_data->local_mass[local_patch * n_dofs_2d * dim * dim +
                                        component * n_dofs_2d * dim],
               &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                           component * n_dofs_2d * dim],
               &shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) +
                                 component * n_dofs_component * (dim - 1)]);
    __syncthreads();

    // B * P
    for (unsigned int i = 0; i < n_dofs_component / block_size + 1; ++i)
      if (tid_g + i * block_size < n_dofs_component)
        {
          shared_data
            ->local_src[local_patch * n_patch_dofs + tid_g + i * block_size] =
            shared_data
              ->local_src[local_patch * n_patch_dofs + dim * n_dofs_component +
                          tid_g + i * block_size];
        }
    __syncthreads();

    for (unsigned int i = 0; i < n_dofs_component / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_dofs_component && component != 0)
        {
          shared_data->local_src[local_patch * n_patch_dofs +
                                 component * n_dofs_component +
                                 mapping[tid + i * n_dofs_2d]] =
            shared_data->local_src[local_patch * n_patch_dofs +
                                   gpu_data->ltoh_dgn_p[tid + i * n_dofs_2d]];
        }
    __syncthreads();

    eval.template vmult_mixed<shapeB, shapeP, false, false, true>(
      &shared_data
         ->local_dst[local_patch * n_patch_dofs + component * n_dofs_component],
      &shared_data
         ->local_src[local_patch * n_patch_dofs + component * n_dofs_component],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * dim * (dim - 1) +
                                   component * n_dofs_2d * (dim - 1)],
      &shared_data
         ->local_mix_der[local_patch * n_dofs_2d * dim + component * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs_tmp * (dim - 1) +
                        component * n_dofs_component * (dim - 1)]);
    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant laplace,
            typename SharedData>
  __device__ void
  evaluate_residual(const unsigned int local_patch, SharedData *shared_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;

    const int tid_y = threadIdx.y % n_dofs_1d;
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    TPEvaluatorStokes<laplace, Number, fe_degree, dim> eval;
    __syncthreads();

    using shapeB = Shape<n_dofs_1d, n_dofs_1d - 1>;
    using shapeU = Shape<n_dofs_1d - 1, n_dofs_1d, n_dofs_1d - 1>;
    using shapeP = Shape<n_dofs_1d - 1, n_dofs_1d - 1, n_dofs_1d - 1>;

    // B^T * U
    eval.template vmult_mixed<shapeB, shapeU, true, false, false, true>(
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim],
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      shared_data->local_mix_mass,
      &shared_data->local_mix_der[local_patch * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        n_patch_dofs_rt + component * n_patch_dofs_rt / dim]);
    __syncthreads();

    const unsigned int *mapping = component == 0 ? ltoh_dgn :
                                  component == 1 ? ltoh_dgt :
                                                   ltoh_dgz;
    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg)
        {
          atomicAdd(&shared_data->local_src[local_patch * n_patch_dofs +
                                            n_patch_dofs_rt +
                                            ltoh_dgn[tid + i * n_dofs_2d]],
                    -shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                                      component * n_patch_dofs_rt / dim +
                                      mapping[tid + i * n_dofs_2d]]);
        }
    __syncthreads();

    // M * U
    eval.template vmult<true>(
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_mass[local_patch * n_dofs_2d * dim * dim],
      &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                  component * n_dofs_2d * dim],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim * (dim - 1)]);
    __syncthreads();

    // B * P
    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg)
        {
          shared_data->local_dst[local_patch * n_patch_dofs +
                                 component * n_patch_dofs_dg +
                                 mapping[tid + i * n_dofs_2d]] =
            -shared_data
               ->local_dst[local_patch * n_patch_dofs + n_patch_dofs_rt +
                           ltoh_dgn[tid + i * n_dofs_2d]];
        }
    __syncthreads();

    eval.template vmult_mixed<shapeB, shapeP, false, false, true, true>(
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data
         ->local_dst[local_patch * n_patch_dofs + component * n_patch_dofs_dg],
      shared_data->local_mix_mass,
      &shared_data->local_mix_der[local_patch * n_dofs_2d * dim],
      &shared_data->tmp[local_patch * n_patch_dofs * (dim - 1) +
                        component * n_patch_dofs_rt / dim]);
    __syncthreads();
  }

  template <int dim, int fe_degree, typename Number, typename SharedData>
  __device__ void
  evaluate_smooth_p(const unsigned int local_patch, SharedData *shared_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;

    const int tid_y = threadIdx.y % n_dofs_1d;
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    TPEvaluatorStokes<LaplaceVariant::Basic, Number, fe_degree, dim> eval;
    __syncthreads();

    eval.inverse(
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * dim * dim +
                               component * n_dofs_1d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                  component * n_dofs_2d * dim],
      &shared_data->tmp[local_patch * n_patch_dofs * 2 +
                        component * n_patch_dofs_rt / dim]);
    __syncthreads();

    using shapeB = Shape<2 * fe_degree + 1, 2 * fe_degree + 2>;
    using shapeU =
      Shape<2 * fe_degree + 2, 2 * fe_degree + 1, 2 * fe_degree + 2>;

    // B^T * U
    eval.template vmult_mixed<shapeB, shapeU, true, false, false, true>(
      &shared_data
         ->tmp[local_patch * n_patch_dofs * 2 + component * n_patch_dofs_dg],
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * (dim - 1)],
      &shared_data->local_mix_der[local_patch * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs * 2 + dim * n_patch_dofs_dg +
                        component * n_patch_dofs_dg]);
    __syncthreads();

    const unsigned int *mapping = component == 0 ? ltoh_dgn :
                                  component == 1 ? ltoh_dgt :
                                                   ltoh_dgz;
    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg)
        {
          atomicAdd(&shared_data->local_src[local_patch * n_patch_dofs +
                                            n_patch_dofs_rt +
                                            ltoh_dgn[tid + i * n_dofs_2d]],
                    shared_data->tmp[local_patch * n_patch_dofs * 2 +
                                     component * n_patch_dofs_dg +
                                     mapping[tid + i * n_dofs_2d]]);
        }
    __syncthreads();
  }

  template <int dim, int fe_degree, typename Number, typename SharedData>
  __device__ void
  evaluate_smooth_u(const unsigned int local_patch, SharedData *shared_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    TPEvaluatorStokes<LaplaceVariant::Basic, Number, fe_degree, dim> eval;
    __syncthreads();

    using shapeB = Shape<2 * fe_degree + 1, 2 * fe_degree + 2>;
    using shapeP =
      Shape<2 * fe_degree + 2, 2 * fe_degree + 2, 2 * fe_degree + 2>;

    eval.template vmult_mixed<shapeB, shapeP, false, false, false, true>(
      &shared_data->tmp[local_patch * n_patch_dofs * 2 +
                        component * n_patch_dofs_rt / dim],
      &shared_data
         ->local_src[local_patch * n_patch_dofs + component * n_patch_dofs_dg],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * (dim - 1)],
      &shared_data->local_mix_der[local_patch * n_dofs_2d],
      &shared_data->tmp[local_patch * n_patch_dofs * 2 + n_patch_dofs_rt +
                        component * n_patch_dofs_dg]);
    __syncthreads();

    eval.template inverse<true>(
      &shared_data->local_dst[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim],
      &shared_data->tmp[local_patch * n_patch_dofs * 2 +
                        component * n_patch_dofs_rt / dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * dim * dim +
                               component * n_dofs_1d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                  component * n_dofs_2d * dim],
      &shared_data->local_src[local_patch * n_patch_dofs +
                              component * n_patch_dofs_rt / dim]);
    __syncthreads();
  }

  template <int dim, int fe_degree, typename Number, typename SharedData>
  __device__ void
  schur_vmult(const unsigned int local_patch,
              SharedData        *shared_data,
              Number            *src,
              Number            *dst,
              Number            *tmp)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;
    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);

    const int component = (threadIdx.y / n_dofs_1d) % dim;

    const int tid_y = threadIdx.y % n_dofs_1d;
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    TPEvaluatorStokes<LaplaceVariant::Basic, Number, fe_degree, dim> eval;

    const unsigned int *mapping = component == 0 ? ltoh_dgn :
                                  component == 1 ? ltoh_dgt :
                                                   ltoh_dgz;

    for (int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg && component != 0)
        {
          src[component * n_patch_dofs_dg + mapping[tid + i * n_dofs_2d]] =
            src[ltoh_dgn[tid + i * n_dofs_2d]];
        }
    __syncthreads();

    using shapeB = Shape<2 * fe_degree + 1, 2 * fe_degree + 2>;
    using shapeU =
      Shape<2 * fe_degree + 2, 2 * fe_degree + 1, 2 * fe_degree + 2>;
    using shapeP =
      Shape<2 * fe_degree + 2, 2 * fe_degree + 2, 2 * fe_degree + 2>;

    // B * src
    eval.template vmult_mixed<shapeB, shapeP, false, false, false, true>(
      &dst[component * n_patch_dofs_rt / dim],
      &src[component * n_patch_dofs_dg],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * (dim - 1)],
      &shared_data->local_mix_der[local_patch * n_dofs_2d],
      &tmp[component * n_patch_dofs_dg]);
    __syncthreads();

    // M^-1 * B * src
    eval.template inverse<false>(
      &src[n_patch_dofs_dg + component * n_patch_dofs_rt / dim],
      &dst[component * n_patch_dofs_rt / dim],
      &shared_data->local_mass[local_patch * n_dofs_1d * dim * dim +
                               component * n_dofs_1d * dim],
      &shared_data->local_laplace[local_patch * n_dofs_2d * dim * dim +
                                  component * n_dofs_2d * dim],
      &tmp[component * n_patch_dofs_dg]);
    __syncthreads();

    // B^T * M^-1 * B * src
    eval.template vmult_mixed<shapeB, shapeU, true, false, false, true>(
      &dst[component * n_patch_dofs_dg],
      &src[n_patch_dofs_dg + component * n_patch_dofs_rt / dim],
      &shared_data->local_mix_mass[local_patch * n_dofs_2d * (dim - 1)],
      &shared_data->local_mix_der[local_patch * n_dofs_2d],
      &tmp[component * n_patch_dofs_dg]);
    __syncthreads();

    for (unsigned int i = 0; i < n_patch_dofs_dg / n_dofs_2d + 1; ++i)
      if (tid + i * n_dofs_2d < n_patch_dofs_dg && component != 0)
        {
          atomicAdd(
            &dst[ltoh_dgn[tid + i * n_dofs_2d]],
            dst[component * n_patch_dofs_dg + mapping[tid + i * n_dofs_2d]]);
        }
    __syncthreads();
  }

  template <int matrix_dim, typename Number>
  __device__ void
  innerProd(const int    &tid,
            const int    &block_size,
            const Number *v1,
            const Number *v2,
            Number       *result)
  {
    if (tid == 0)
      *result = 0;
    __syncthreads();

    Number sum = 0;
    for (unsigned int i = 0; i < matrix_dim / block_size + 1; ++i)
      if (tid + i * block_size < matrix_dim)
        {
          sum += v1[tid + i * block_size] * v2[tid + i * block_size];
        }

    sum += __shfl_down_sync(-1u, sum, 1);
    sum += __shfl_down_sync(-1u, sum, 2);
    sum += __shfl_down_sync(-1u, sum, 4);
    sum += __shfl_down_sync(-1u, sum, 8);
    sum += __shfl_down_sync(-1u, sum, 16);

    if ((tid % 32) == 0)
      atomicAdd(result, sum);
  }

  template <int matrix_dim, typename Number, bool self_scaling>
  __device__ void
  VecSadd(const int &tid,
          const int &block_size,
          Number    *v1,
          Number    *v2,
          Number     alpha)
  {
    for (unsigned int i = 0; i < matrix_dim / block_size + 1; ++i)
      if (tid + i * block_size < matrix_dim)
        {
          if (self_scaling)
            v1[tid + i * block_size] =
              alpha * v1[tid + i * block_size] + v2[tid + i * block_size];
          else
            v1[tid + i * block_size] += alpha * v2[tid + i * block_size];
        }
  }

  template <int dim, int fe_degree, typename Number, typename SharedData>
  __device__ void
  evaluate_smooth_cg(const unsigned int local_patch, SharedData *shared_data)
  {
    constexpr int shift     = Util::pow(fe_degree + 1, dim);
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_1d * n_dofs_1d * dim;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    TPEvaluatorStokes<LaplaceVariant::Basic, Number, fe_degree, dim> eval;

    Number *x =
      &shared_data->local_dst[local_patch * n_patch_dofs + n_patch_dofs_rt];
    Number *p  = &shared_data->local_src[local_patch * n_patch_dofs];
    Number *Ap = &shared_data->tmp[local_patch * n_patch_dofs * 4];
    Number *r =
      &shared_data->tmp[local_patch * n_patch_dofs * 2 + n_patch_dofs];
    Number *tmp =
      &shared_data
         ->tmp[local_patch * n_patch_dofs * 2 + n_patch_dofs + n_patch_dofs_dg];
    __syncthreads();

    for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
      if (tid + i * block_size < n_patch_dofs_dg)
        {
          r[tid + i * block_size] = p[tid + i * block_size + n_patch_dofs_rt];
          p[tid + i * block_size] = p[tid + i * block_size + n_patch_dofs_rt];
          x[tid + i * block_size] = 0.;
        }

    constexpr int MAX_IT = 30;

    Number *rsold    = &shared_data->local_vars[7 * local_patch + 0];
    Number *norm_min = &shared_data->local_vars[7 * local_patch + 1];
    Number *norm_act = &shared_data->local_vars[7 * local_patch + 2];

    Number *alpha = &shared_data->local_vars[7 * local_patch + 3];
    Number *beta  = &shared_data->local_vars[7 * local_patch + 4];

    Number *rsnew = &shared_data->local_vars[7 * local_patch + 5];

    Number *convergenced = &shared_data->local_vars[6];
    Number  local_flag   = -1;

    if (threadIdx.x == 0 && threadIdx.y == 0)
      *convergenced = 1;

    innerProd<n_patch_dofs_dg, Number>(tid, block_size, r, r, rsold);
    __syncthreads();

    if (tid == 0)
      atomicAdd(convergenced, -2);

    if (tid == 0)
      {
        *norm_min = sqrt(*rsold);
        *norm_act = sqrt(*rsold);
      }
    __syncthreads();

    Number local_norm_min = *norm_min;
    Number local_norm_act = *norm_act;

#ifdef TIMING
    long long int time_schur     = 0;
    long long int time_vec       = 0;
    long long int time_reduction = 0;
#endif

    for (int it = 0; it < MAX_IT; ++it)
      {
#ifdef TIMING
        __syncthreads();
        auto start_s = clock64();
#endif
        schur_vmult<dim, fe_degree, Number, SharedData>(
          local_patch, shared_data, p, Ap, tmp);
        __syncthreads();
#ifdef TIMING
        time_schur += clock64() - start_s;
#endif

#ifdef TIMING
        __syncthreads();
        auto start_r = clock64();
#endif
        innerProd<n_patch_dofs_dg, Number>(tid, block_size, p, Ap, alpha);
        __syncthreads();
#ifdef TIMING
        time_reduction += clock64() - start_r;
#endif

        if (tid == 0)
          *alpha = *rsold / *alpha;
        __syncthreads();

#ifdef TIMING
        __syncthreads();
        auto start_v = clock64();
#endif
        VecSadd<n_patch_dofs_dg, Number, false>(
          tid, block_size, r, Ap, -*alpha);
        __syncthreads();
#ifdef TIMING
        time_vec += clock64() - start_v;
#endif

#ifdef TIMING
        __syncthreads();
        auto start_r1 = clock64();
#endif
        innerProd<n_patch_dofs_dg, Number>(tid, block_size, r, r, rsnew);
        __syncthreads();
#ifdef TIMING
        time_reduction += clock64() - start_r1;
#endif

        if (tid == 0)
          *norm_act = sqrt(*rsnew);
        __syncthreads();

        local_norm_act = *norm_act;

        if (local_norm_act < local_norm_min)
          {
            if (tid == 0)
              {
                *norm_min = *norm_act;
              }

            local_norm_min = local_norm_act;
          }
        else if (local_flag < 0 &&
                 (local_norm_act >= local_norm_min || fabs(*alpha) < 1e-10))
          {
#ifdef PRINT_INFO
            if (tid == 0 && blockIdx.x == 0)
              printf("Converged 2. # it: %d, residual: %e\n", it, *norm_min);
#endif
            VecSadd<n_patch_dofs_dg, Number, true>(
              tid,
              block_size,
              &shared_data->local_src[local_patch * n_patch_dofs],
              x,
              0);

#ifdef TIMING
            if (tid == 0 && blockIdx.x == 0)
              printf("Time info: schur: %lld, vec: %lld, innerprod: %lld\n",
                     time_schur,
                     time_vec,
                     time_reduction);
#endif
            // return;
            local_flag = 1;
            if (tid == 0)
              atomicAdd(convergenced, 2);
          }

#ifdef TIMING
        __syncthreads();
        auto start_v1 = clock64();
#endif
        VecSadd<n_patch_dofs_dg, Number, false>(tid, block_size, x, p, *alpha);
        __syncthreads();
#ifdef TIMING
        time_vec += clock64() - start_v1;
#endif

        if (local_flag < 0 && *norm_min < 1e-12)
          {
#ifdef PRINT_INFO
            if (tid == 0 && blockIdx.x == 0)
              printf("Converged 1. # it: %d, residual: %e\n", it, *norm_min);
#endif
            VecSadd<n_patch_dofs_dg, Number, true>(
              tid,
              block_size,
              &shared_data->local_src[local_patch * n_patch_dofs],
              x,
              0);

#ifdef TIMING
            if (tid == 0 && blockIdx.x == 0)
              printf("Time info: schur: %lld, vec: %lld, innerprod: %lld\n",
                     time_schur,
                     time_vec,
                     time_reduction);
#endif
            // return;
            local_flag = 1;
            if (tid == 0)
              atomicAdd(convergenced, 2);
          }

        if (tid == 0)
          *beta = *rsnew / *rsold;
        __syncthreads();

        if (*convergenced > 0)
          return;

#ifdef TIMING
        __syncthreads();
        auto start_v2 = clock64();
#endif
        VecSadd<n_patch_dofs_dg, Number, true>(tid, block_size, p, r, *beta);
        __syncthreads();
#ifdef TIMING
        time_vec += clock64() - start_v2;
#endif
        if (tid == 0)
          *rsold = *rsnew;
      }
#ifdef PRINT_INFO
    if (tid == 0 && blockIdx.x == 0)
      printf("Converged 0. # it: %d, residual: %e\n", MAX_IT, *norm_min);
#endif
  }

  // template <int dim,
  //           int fe_degree,
  //           typename Number,
  //           LaplaceVariant     laplace,
  //           LocalSolverVariant solver>
  // __device__ void
  // evaluate_smooth_cf(
  //   const unsigned int local_patch,
  //   SharedDataSmoother<dim, Number, SmootherVariant::ConflictFree, solver>
  //                                                                 *shared_data,
  //   const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  // {
  //   constexpr unsigned int n_dofs_1d   = 2 * fe_degree + 1;
  //   constexpr unsigned int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 2;
  //   constexpr unsigned int local_dim   = Util::pow(n_dofs_1d, dim);

  //   TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
  //   TPEvaluatorSmootherInv<Number,
  //                          n_dofs_1d,
  //                          SmootherVariant::ConflictFree,
  //                          solver,
  //                          dim>
  //     eval_inverse;
  //   __syncthreads();

  //   eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
  //                    &shared_data->local_dst[local_patch * local_dim],
  //                    shared_data->local_mass,
  //                    shared_data->local_laplace,
  //                    shared_data->local_bilaplace,
  //                    &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
  //   __syncthreads();

  //   const unsigned int linear_tid =
  //     threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

  //   if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //     {
  //       unsigned int row = linear_tid / (n_dofs_1d - 2);
  //       unsigned int col = linear_tid % (n_dofs_1d - 2);

  //       if (row < dim)
  //         shared_data->local_mass[row * (n_dofs_1d - 2) + col] =
  //           gpu_data->eigenvalues[row * (n_dofs_1d - 2) + col];

  //       for (unsigned int d = 0; d < dim; ++d)
  //         shared_data
  //           ->local_laplace[(d * (n_dofs_1d - 2) + row) * (n_dofs_1d - 2) +
  //                           col] =
  //           gpu_data
  //             ->eigenvectors[(d * (n_dofs_1d - 2) + row) * (n_dofs_1d - 2) +
  //                            col];
  //     }
  //   // __syncthreads();


  //   if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //     {
  //       unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
  //       unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

  //       for (unsigned int z = 0; z < n_dofs_1d_z; ++z)
  //         {
  //           shared_data
  //             ->tmp[2 * local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
  //                   (row - 1) * (n_dofs_1d - 2) + col - 1] =
  //             shared_data->local_dst[local_patch * local_dim +
  //                                    (z + dim - 2) * n_dofs_1d * n_dofs_1d +
  //                                    row * n_dofs_1d + col];

  //           shared_data->tmp[2 * local_patch * local_dim + local_dim +
  //                            z * n_dofs_1d * n_dofs_1d +
  //                            (row - 1) * (n_dofs_1d - 2) + col - 1] =
  //             shared_data->local_src[local_patch * local_dim +
  //                                    (z + dim - 2) * n_dofs_1d * n_dofs_1d +
  //                                    row * n_dofs_1d + col];
  //         }
  //     }
  //   __syncthreads();

  //   eval_inverse.apply_inverse(
  //     &shared_data->tmp[local_patch * local_dim * 2],
  //     &shared_data->tmp[local_patch * local_dim * 2 + local_dim],
  //     shared_data->local_mass,
  //     shared_data->local_laplace,
  //     &shared_data->local_src[local_patch * local_dim]);
  //   __syncthreads();
  // }


  // template <int dim, int fe_degree, typename Number, LocalSolverVariant
  // solver>
  // __device__ void
  // evaluate_smooth_global(
  //   const unsigned int local_patch,
  //   SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, solver>
  //     *shared_data)
  // {
  //   constexpr unsigned int n_dofs_1d = 2 * fe_degree - 1;
  //   constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

  //   TPEvaluatorSmootherInv<Number,
  //                          n_dofs_1d,
  //                          SmootherVariant::GLOBAL,
  //                          solver,
  //                          dim>
  //     eval;
  //   __syncthreads();

  //   eval.apply_inverse(
  //     &shared_data->local_dst[local_patch * local_dim],
  //     &shared_data->local_src[local_patch * local_dim],
  //     &shared_data->local_mass[local_patch * n_dofs_1d * dim],
  //     &shared_data->local_laplace[local_patch * n_dofs_1d * n_dofs_1d * dim],
  //     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
  //   __syncthreads();
  // }

} // namespace PSMF


#endif // CUDA_EVALUATE_CUH
