/**
 * @file loop_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of global functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef LOOP_KERNEL_CUH
#define LOOP_KERNEL_CUH

#include <type_traits>

#include "evaluate_kernel.cuh"
#include "patch_base.cuh"

namespace PSMF
{

  /**
   * Shared data for laplace operator kernel.
   * We have to use the following thing to avoid
   * a compile time error since @tparam Number
   * could be double and float at same time.
   */
  extern __shared__ double data_d[];
  extern __shared__ float  data_f[];

  template <typename Number>
  __device__ inline Number *
  get_shared_data_ptr();

  template <>
  __device__ inline double *
  get_shared_data_ptr()
  {
    return data_d;
  }

  template <>
  __device__ inline float *
  get_shared_data_ptr()
  {
    return data_f;
  }

  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_matrix(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr unsigned int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr unsigned int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;

    const unsigned int tid   = threadIdx.x;
    const unsigned int patch = blockIdx.x;

    SharedDataOp<dim, Number, LaplaceVariant::MatrixStruct> shared_data(
      get_shared_data_ptr<Number>(), 1, 0, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // if (tid == 0)
        //   printf("%d ", patch);

        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_laplace[patch * n_patch_dofs + tid +
                                           i * blockDim.x];

              shared_data.local_src[tid + i * blockDim.x] =
                src[global_dof_index];
              shared_data.local_dst[tid + i * blockDim.x] = 0;
            }

        __syncthreads();

        constexpr unsigned int matrix_size = Util::pow(n_patch_dofs, 2);
        unsigned int           patch_type  = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        for (unsigned int row = 0; row < n_patch_dofs; ++row) // 与此前相同；前后两个循环读/写数据，中间这里调用NN kernel
          {
            for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs)
                {
                  auto val =
                    gpu_data.vertex_patch_matrices[patch_type * matrix_size +
                                                   row * n_patch_dofs + tid +
                                                   i * blockDim.x] *
                    shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_dst[row], val);
                }
          }
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_laplace[patch * n_patch_dofs + tid +
                                           i * blockDim.x];

              atomicAdd(&dst[global_dof_index],
                        shared_data.local_dst[tid + i * blockDim.x]);
            }
      }
  }


  template <int dim, int fe_degree, typename Number>
  __global__ void
  laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int block_size      = n_dofs_2d * dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    SharedDataOp<dim, Number, LaplaceVariant::Basic> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        // L M
        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < n_dofs_2d) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                auto dd = dir == 0 ? d :
                          dir == 1 ? (d == 0 ? 1 :
                                      d == 1 ? 0 :
                                               2) :
                                     (d == 0 ? 2 :
                                      d == 1 ? 0 :
                                               1);

                const int shift = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_mass[shift + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data.rt_mass_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                           n_dofs_2d +
                                         tid];
                shared_data
                  .local_laplace[shift + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data
                    .rt_laplace_1d[d][gpu_data.patch_type[patch * dim + dd] *
                                        n_dofs_2d +
                                      tid];
              }

        for (int dir = 0; dir < dim; ++dir)
          {
            // D
            const int shift1 = local_patch * n_dofs_2d * dim;
            if (tid < n_dofs_1d * (n_dofs_1d - 1))
              shared_data.local_mix_der[shift1 + dir * n_dofs_2d + tid] =
                -gpu_data.mix_der_1d[0][gpu_data.patch_type[patch * dim + dir] *
                                          n_dofs_2d +
                                        tid];
            // M
            const int shift2 = local_patch * n_dofs_2d * dim * (dim - 1);
            auto      dd1    = dir == 0 ? 1 : 0;
            if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
              shared_data
                .local_mix_mass[shift2 + dir * n_dofs_2d * (dim - 1) + tid] =
                gpu_data.mix_mass_1d[1][gpu_data.patch_type[patch * dim + dd1] *
                                          n_dofs_2d +
                                        tid];
            if (dim == 3)
              {
                auto dd2 = dir == 2 ? 1 : 2;
                if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                  shared_data
                    .local_mix_mass[shift2 + dir * n_dofs_2d * (dim - 1) +
                                    n_dofs_2d + tid] =
                    gpu_data
                      .mix_mass_1d[2][gpu_data.patch_type[patch * dim + dd2] *
                                        n_dofs_2d +
                                      tid];
              }
          }


        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_laplace[patch * n_patch_dofs +
                                           htol_rt[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
            }
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index_n =
                gpu_data
                  .patch_dof_laplace[patch * n_patch_dofs + n_patch_dofs_rt +
                                     htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                src[global_dof_index_n];
            }


        evaluate_laplace<dim, fe_degree, Number, LaplaceVariant::Basic>(
          local_patch, &shared_data);
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_laplace[patch * n_patch_dofs +
                                           htol_rt[tid + i * block_size]];

              atomicAdd(&dst[global_dof_index],
                        shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                              i * block_size]);
            }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_laplace[patch * n_patch_dofs + n_patch_dofs_rt +
                                     tid + i * block_size];

              atomicAdd(
                &dst[global_dof_index],
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size]);
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::Direct>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3) +
      Util::pow(2 * fe_degree + 2, dim);

    const unsigned int tid   = threadIdx.x;
    const unsigned int patch = blockIdx.x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(), 1, 0, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              shared_data.local_src[tid + i * blockDim.x] =
                src[global_dof_index];
              shared_data.local_dst[tid + i * blockDim.x] =
                dst[global_dof_index];
            }
        __syncthreads();

        constexpr unsigned int matrix_size = Util::pow(n_patch_dofs, 2);
        unsigned int           patch_type  = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        for (unsigned int row = 0; row < n_patch_dofs; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs)
                {
                  auto val =
                    gpu_data
                      .eigenvalues[patch_type * matrix_size +
                                   row * n_patch_dofs + tid + i * blockDim.x] *
                    shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_dst[row], val);
                }
          }
        __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              dst[global_dof_index] =
                shared_data.local_dst[tid + i * blockDim.x] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__ typename std::enable_if<local_solver ==
                                     LocalSolverVariant::SchurDirect>::type
  loop_kernel_global(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3);

    constexpr unsigned int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3) +
      Util::pow(2 * fe_degree + 2, dim);

    const unsigned int tid   = threadIdx.x;
    const unsigned int patch = blockIdx.x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(), 1, 0, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              shared_data.local_src[tid + i * blockDim.x] =
                src[global_dof_index];
              shared_data.local_dst[tid + i * blockDim.x] =
                dst[global_dof_index];
              shared_data.tmp[tid + i * blockDim.x] = 0;
            }
        __syncthreads();

        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        constexpr unsigned int matrix_size   = Util::pow(n_patch_dofs, 2);
        constexpr unsigned int matrix_size_U = Util::pow(n_patch_dofs_rt, 2);
        constexpr unsigned int matrix_size_P = Util::pow(n_patch_dofs_dg, 2);

        /// B^T M^-1 B P = B^T M^-1 F - G
        // M^-1 F
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val = gpu_data.eigenvalues[patch_type * matrix_size +
                                                  row * n_patch_dofs_rt + tid +
                                                  i * blockDim.x] *
                             shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.tmp[row], val);
                }
          }
        __syncthreads();
        // B^T M^-1 F - G
        for (unsigned int row = 0; row < n_patch_dofs_dg; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val =
                    -gpu_data.eigenvalues[patch_type * matrix_size +
                                          matrix_size_U + matrix_size_P +
                                          n_patch_dofs_rt * n_patch_dofs_dg +
                                          row * n_patch_dofs_rt + tid +
                                          i * blockDim.x] *
                    shared_data.tmp[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_src[n_patch_dofs_rt + row], val);
                }
          }
        __syncthreads();
        // B^T M^-1 B P = B^T M^-1 F - G
        for (unsigned int row = 0; row < n_patch_dofs_dg; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_dg)
                {
                  auto val =
                    -gpu_data
                       .eigenvalues[patch_type * matrix_size + matrix_size_U +
                                    row * n_patch_dofs_dg + tid +
                                    i * blockDim.x] *
                    shared_data
                      .local_src[n_patch_dofs_rt + tid + i * blockDim.x];

                  atomicAdd(&shared_data.tmp[n_patch_dofs_rt + row], val);
                }
          }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs_dg)
            shared_data.local_dst[n_patch_dofs_rt + tid + i * blockDim.x] +=
              shared_data.tmp[n_patch_dofs_rt + tid + i * blockDim.x];

        /// M U = F - B P
        // F - B P
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_dg)
                {
                  auto val =
                    -gpu_data
                       .eigenvalues[patch_type * matrix_size + matrix_size_U +
                                    matrix_size_P + row * n_patch_dofs_dg +
                                    tid + i * blockDim.x] *
                    shared_data.tmp[n_patch_dofs_rt + tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_src[row], val);
                }
          }
        __syncthreads();
        // M U = F - B P
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val = gpu_data.eigenvalues[patch_type * matrix_size +
                                                  row * n_patch_dofs_rt + tid +
                                                  i * blockDim.x] *
                             shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_dst[row], val);
                }
          }
        __syncthreads();
        // store
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              dst[global_dof_index] =
                shared_data.local_dst[tid + i * blockDim.x] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver ==
                            LocalSolverVariant::SchurTensorProduct>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int n_patch_dofs_rt_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);
    constexpr int block_size = n_dofs_2d * dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        // eigs
        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < (n_dofs_1d - 2) * (n_dofs_1d - 1)) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                if ((d == 0 && tid < n_dofs_1d - 2) ||
                    (d != 0 && tid < n_dofs_1d - 1))
                  {
                    const int shift = local_patch * n_dofs_1d * dim * dim;
                    shared_data
                      .local_mass[shift + (dir * dim + d) * n_dofs_1d + tid] =
                      gpu_data.eigvals[dir][patch_type * n_dofs_1d * dim +
                                            d * n_dofs_1d + tid];
                  }
                const int shift2 = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_laplace[shift2 + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data.eigvecs[dir][patch_type * n_dofs_2d * dim +
                                        d * n_dofs_2d + tid];
              }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * 1;
          if (tid < (n_dofs_1d - 2) * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + tid] =
              -gpu_data.smooth_mixder_1d[0][tid];
          // M
          const int shift2 = local_patch * n_dofs_2d * (dim - 1);
          if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
            shared_data.local_mix_mass[shift2 + tid] =
              gpu_data.smooth_mixmass_1d[1][tid];

          if (dim == 3)
            {
              if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                shared_data.local_mix_mass[shift2 + n_dofs_2d + tid] =
                  gpu_data.smooth_mixmass_1d[2][tid];
            }
        }

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    htol_rt_interior[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
              shared_data
                .local_dst[local_patch * n_patch_dofs + tid + i * block_size] =
                0;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          n_patch_dofs_rt_all +
                                          htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                -src[global_dof_index];
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] = 0;
            }
        __syncthreads();

        /// B^T M^-1 B P = B^T M^-1 F - G
        // B^T M^-1 F - G
        evaluate_smooth_p<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        evaluate_smooth_cg<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        // for (unsigned int row = 0; row < n_patch_dofs_dg; ++row)
        //   {
        //     for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1;
        //     ++i)
        //       if (tid + i * block_size < n_patch_dofs_dg)
        //         {
        //           auto val =
        //             -gpu_data.inverse_schur[patch_type * matrix_size_P +
        //                                     row * n_patch_dofs_dg + tid +
        //                                     i * block_size] *
        //             shared_data
        //               .local_src[local_patch * n_patch_dofs + n_patch_dofs_rt
        //               +
        //                          tid + i * block_size];

        //           atomicAdd(&shared_data.tmp[local_patch * n_patch_dofs +
        //                                      n_patch_dofs_rt + row],
        //                     val);
        //         }
        //   }
        // __syncthreads();

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + n_patch_dofs_dg +
                           ltoh_dgt[tid + i * block_size]] =
                shared_data.local_src[local_patch * n_patch_dofs +
                                      ltoh_dgn[tid + i * block_size]];

              if constexpr (dim == 3)
                shared_data
                  .local_src[local_patch * n_patch_dofs + 2 * n_patch_dofs_dg +
                             ltoh_dgz[tid + i * block_size]] =
                  shared_data.local_src[local_patch * n_patch_dofs +
                                        ltoh_dgn[tid + i * block_size]];
            }

        /// M U = F - B P
        evaluate_smooth_u<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        // store
        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    htol_rt_interior[tid + i * block_size]];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                      i * block_size] *
                gpu_data.relaxation;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    n_patch_dofs_rt_all + tid + i * block_size];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs +
                                      n_patch_dofs_rt + tid + i * block_size] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::Uzawa>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 3;
    constexpr int n_dofs_2d = n_dofs_1d * n_dofs_1d;

    constexpr int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 1);
    constexpr int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);
    constexpr int n_patch_dofs    = n_patch_dofs_rt + n_patch_dofs_dg;
    constexpr int n_patch_dofs_rt_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);
    constexpr int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);
    constexpr int block_size = n_dofs_2d * dim;

    const int patch_per_block = gpu_data.patch_per_block;
    const int local_patch     = threadIdx.y / (n_dofs_1d * dim);
    const int patch           = local_patch + patch_per_block * blockIdx.x;

    const int tid_y = threadIdx.y % (n_dofs_1d * dim);
    const int tid_x = threadIdx.x;
    const int tid   = tid_y * n_dofs_1d + tid_x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(),
                  patch_per_block,
                  n_dofs_1d,
                  n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        // eigs
        for (int dir = 0; dir < dim; ++dir)
          for (int d = 0; d < dim; ++d)
            if ((d == 0 && tid < (n_dofs_1d - 2) * (n_dofs_1d - 1)) ||
                (d != 0 && tid < (n_dofs_1d - 1) * (n_dofs_1d - 1)))
              {
                if ((d == 0 && tid < n_dofs_1d - 2) ||
                    (d != 0 && tid < n_dofs_1d - 1))
                  {
                    const int shift = local_patch * n_dofs_1d * dim * dim;
                    shared_data
                      .local_mass[shift + (dir * dim + d) * n_dofs_1d + tid] =
                      gpu_data.eigvals[dir][patch_type * n_dofs_1d * dim +
                                            d * n_dofs_1d + tid];
                  }
                const int shift2 = local_patch * n_dofs_2d * dim * dim;
                shared_data
                  .local_laplace[shift2 + (dir * dim + d) * n_dofs_2d + tid] =
                  gpu_data.eigvecs[dir][patch_type * n_dofs_2d * dim +
                                        d * n_dofs_2d + tid];
              }

        {
          // D
          const int shift1 = local_patch * n_dofs_2d * 1;
          if (tid < (n_dofs_1d - 2) * (n_dofs_1d - 1))
            shared_data.local_mix_der[shift1 + tid] =
              -gpu_data.smooth_mixder_1d[0][tid];
          // M
          const int shift2 = local_patch * n_dofs_2d * (dim - 1);
          if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
            shared_data.local_mix_mass[shift2 + tid] =
              gpu_data.smooth_mixmass_1d[1][tid];

          if (dim == 3)
            {
              if (tid < (n_dofs_1d - 1) * (n_dofs_1d - 1))
                shared_data.local_mix_mass[shift2 + n_dofs_2d + tid] =
                  gpu_data.smooth_mixmass_1d[2][tid];
            }
        }

        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    htol_rt_interior[tid + i * block_size]];

              shared_data
                .local_src[local_patch * n_patch_dofs + tid + i * block_size] =
                src[global_dof_index];
              shared_data
                .local_dst[local_patch * n_patch_dofs + tid + i * block_size] =
                0;
            }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          n_patch_dofs_rt_all +
                                          htol_dgn[tid + i * block_size]];

              shared_data.local_src[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] =
                src[global_dof_index];
              shared_data.local_dst[local_patch * n_patch_dofs +
                                    n_patch_dofs_rt + tid + i * block_size] = 0;
            }
        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     for (unsigned int i = 0; i < n_patch_dofs; ++i)
        //       printf("%f ", shared_data.local_src[i]);
        //     printf("src Uzawa\n\n");
        //   }
        //
        // __syncthreads();
        // printf("[%d] ", tid);

        evaluate_smooth_Uzawa<dim, fe_degree, Number, decltype(shared_data)>(
          local_patch, &shared_data);
        __syncthreads();

        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     for (unsigned int i = 0; i < n_patch_dofs_rt; ++i)
        //       printf("%f ", shared_data.local_dst[i]);
        //
        //     for (unsigned int i = 0; i < n_patch_dofs_dg; ++i)
        //       printf("%f ",
        //              shared_data.local_dst[n_patch_dofs_rt + ltoh_dgn[i]]);
        //     printf("x Uzawa\n\n");
        //   }
        //
        // __syncthreads();
        // printf("(%d) ", tid);

        // store
        for (unsigned int i = 0; i < n_patch_dofs_rt / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_rt)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    htol_rt_interior[tid + i * block_size]];

              dst[global_dof_index] +=
                shared_data.local_dst[local_patch * n_patch_dofs + tid +
                                      i * block_size] *
                gpu_data.relaxation;

              // printf("<%d> ", tid);
              // if (tid == 32)
              //   printf("\n store [[[ %d %d %d %d %f]]]\n",
              //          i,
              //          block_size,
              //          n_patch_dofs_rt,
              //          n_patch_dofs_rt / block_size,
              //          shared_data.local_dst[tid]);
            }

        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     for (unsigned int i = 0; i < n_patch_dofs_rt; ++i)
        //       printf("%d ",
        //              gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
        //                                        htol_rt_interior[i]]);
        //     printf("ind Uzawa\n\n");
        //
        //     for (unsigned int i = 0; i < n_patch_dofs_rt; ++i)
        //       printf("%f ",
        //              dst[gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
        //                                            htol_rt_interior[i]]]);
        //     printf("dstg Uzawa\n\n");
        //   }

        for (unsigned int i = 0; i < n_patch_dofs_dg / block_size + 1; ++i)
          if (tid + i * block_size < n_patch_dofs_dg)
            {
              unsigned int global_dof_index =
                gpu_data
                  .patch_dof_smooth[patch * n_patch_dofs_all +
                                    n_patch_dofs_rt_all + tid + i * block_size];

              dst[global_dof_index] +=
                shared_data
                  .local_dst[local_patch * n_patch_dofs + n_patch_dofs_rt +
                             ltoh_dgn[tid + i * block_size]] *
                gpu_data.relaxation;
            }

        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //   {
        //     __syncthreads();
        //     for (unsigned int i = 0; i < n_patch_dofs_all; ++i)
        //       printf("%f ", dst[i]);
        //     printf("dsta Uzawa\n\n");
        //   }
      }
  }


  template <int matrix_dim, typename Number>
  __device__ void
  MatVecMul(const Number *A, const Number *src, Number *dst)
  {
    const unsigned int tid = threadIdx.x;

    for (unsigned int i = 0; i < matrix_dim / blockDim.x + 1; ++i)
      if (tid + i * blockDim.x < matrix_dim)
        {
          Number tmp = 0;
          for (unsigned int j = 0; j < matrix_dim; ++j)
            tmp += A[(tid + i * blockDim.x) * matrix_dim + j] * src[j];
          dst[tid + i * blockDim.x] = tmp;
        }
  }

  template <int matrix_dim, typename Number>
  __device__ void
  VecDot(const Number *v1, const Number *v2, Number *result)
  {
    const unsigned int tid = threadIdx.x;

    if (tid == 0)
      result[0] = 0;
    __syncthreads();

    for (unsigned int i = 0; i < matrix_dim / blockDim.x + 1; ++i)
      if (tid + i * blockDim.x < matrix_dim)
        {
          auto val = v1[tid + i * blockDim.x] * v2[tid + i * blockDim.x];
          atomicAdd(&result[0], val);
        }
  }

  template <int matrix_dim, typename Number, bool self_scaling>
  __device__ void
  VecOp(Number *v1, Number *v2, Number alpha)
  {
    const unsigned int tid = threadIdx.x;
    for (unsigned int i = 0; i < matrix_dim / blockDim.x + 1; ++i)
      if (tid + i * blockDim.x < matrix_dim)
        {
          if (self_scaling)
            v1[tid + i * blockDim.x] =
              alpha * v1[tid + i * blockDim.x] + v2[tid + i * blockDim.x];
          else
            v1[tid + i * blockDim.x] += alpha * v2[tid + i * blockDim.x];
        }
  }

  template <int matrix_dim, typename Number>
  __device__ void
  solver_CG(const Number *A, Number *x, Number *p, Number *r, Number *tmp)
  {
    const unsigned int tid    = threadIdx.x;
    constexpr int      MAX_IT = 20;
    // constexpr int      n_cells = 1 << 2;

    Number *rsold    = &tmp[2 * matrix_dim + 0];
    Number *norm_min = &tmp[2 * matrix_dim + 1];
    Number *norm_act = &tmp[2 * matrix_dim + 2];

    Number *alpha = &tmp[2 * matrix_dim + 3];
    Number *beta  = &tmp[2 * matrix_dim + 4];

    Number *rsnew = &tmp[2 * matrix_dim + 5];

    VecDot<matrix_dim, Number>(r, r, rsold);

    *norm_min = sqrt(*rsold);
    *norm_act = sqrt(*rsold);

    if (tid == 0)
      {
        // printf("%.3e %.3e %.3e", *rsold, *norm_min, *norm_act);
        // printf("\nDEVICE scaler \n");

        // for (unsigned int i = 0; i < matrix_dim; ++i)
        //   printf("%.3e ", p[i]);
        // printf("\nDEVICE p \n");

        // for (unsigned int i = 0; i < matrix_dim; ++i)
        //   printf("%.3e ", r[i]);
        // printf("\nDEVICE r \n");
      }

    for (unsigned int it = 0; it < MAX_IT; ++it)
      {
        MatVecMul<matrix_dim, Number>(A, p, tmp);
        __syncthreads();

        // if (tid == 0 && it >= 0)
        //   {
        //     for (unsigned int i = 0; i < matrix_dim; ++i)
        //       printf("%.3e ", tmp[i]);
        //     printf("\nDEVICE A*p \n");
        //   }


        VecDot<matrix_dim, Number>(p, tmp, alpha);
        __syncthreads();

        // if (tid == 0)
        //   {
        //     printf("%.3e ", *alpha);
        //     printf("\nDEVICE ALPHA \n");
        //   }

        // if (*alpha < 0)
        //   {
        //     if (tid == 0 && blockIdx.x == 0)
        //       printf("ALPHA # it: %d, residual: %e\n", it, *norm_min);

        //     for (unsigned int c = 0; c < n_cells; ++c)
        //       tmp[matrix_dim + c * matrix_dim / n_cells] = 0;

        //     return;
        //   }

        if (tid == 0)
          *alpha = *rsold / *alpha;
        __syncthreads();

        // if (tid == 0)
        //   {
        //     printf("%.3e ", *alpha);
        //     printf("\nDEVICE ALPHA \n");
        //   }

        VecOp<matrix_dim, Number, false>(x, p, *alpha);
        VecOp<matrix_dim, Number, false>(r, tmp, -*alpha);
        __syncthreads();

        VecDot<matrix_dim, Number>(r, r, rsnew);
        __syncthreads();

        if (tid == 0)
          *norm_act = sqrt(*rsnew);
        __syncthreads();

        // if (tid == 0)
        //   {
        //     printf("%.3e %.3e", *norm_act, *norm_min);
        //     printf("\nDEVICE NORM \n");
        //   }
        __syncthreads();
        if (*norm_act < *norm_min)
          {
            *norm_min = *norm_act;
            VecOp<matrix_dim, Number, true>(&tmp[matrix_dim], x, 0);

            // if (tid == 0)
            //   {
            //     for (unsigned int i = 0; i < matrix_dim; ++i)
            //       printf("%.3e ", tmp[matrix_dim + i]);
            //     printf("\nDEVICE minx \n");
            //   }
          }

        if (*norm_min < 1e-10)
          {
            if (tid == 0 && blockIdx.x == 0)
              printf("# it: %d, residual: %e\n", it, *norm_min);

            // if (tid == 0)
            //   {
            //     for (unsigned int i = 0; i < matrix_dim; ++i)
            //       printf("%.3e ", tmp[matrix_dim + i]);
            //     printf("\nDEVICE minx \n");
            //   }

            return;
          }

        if (tid == 0)
          *beta = *rsnew / *rsold;
        __syncthreads();

        // if (tid == 0)
        //   {
        //     printf("%.3e ", *beta);
        //     printf("\nDEVICE BETA \n");
        //   }

        VecOp<matrix_dim, Number, true>(p, r, *beta);
        __syncthreads();
        *rsold = *rsnew;
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver == LocalSolverVariant::SchurIter>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_patch_dofs_all =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3);

    constexpr unsigned int n_patch_dofs_dg = Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3) +
      Util::pow(2 * fe_degree + 2, dim);

    const unsigned int tid   = threadIdx.x;
    const unsigned int patch = blockIdx.x;

    SharedDataSmoother<dim, Number, SmootherVariant::GLOBAL, local_solver>
      shared_data(get_shared_data_ptr<Number>(), 1, 0, n_patch_dofs);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              shared_data.local_src[tid + i * blockDim.x] =
                src[global_dof_index];
              shared_data.local_dst[tid + i * blockDim.x] =
                dst[global_dof_index];
              shared_data.tmp[tid + i * blockDim.x] = 0;
            }
        __syncthreads();

        unsigned int patch_type = 0;
        for (unsigned int d = 0; d < dim; ++d)
          patch_type += gpu_data.patch_type[patch * dim + d] * Util::pow(3, d);

        constexpr unsigned int matrix_size   = Util::pow(n_patch_dofs, 2);
        constexpr unsigned int matrix_size_U = Util::pow(n_patch_dofs_rt, 2);
        constexpr unsigned int matrix_size_P = Util::pow(n_patch_dofs_dg, 2);

        /// B^T M^-1 B P = B^T M^-1 F - G
        // M^-1 F
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val = gpu_data.eigenvalues[patch_type * matrix_size +
                                                  row * n_patch_dofs_rt + tid +
                                                  i * blockDim.x] *
                             shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.tmp[row], val);
                }
          }
        __syncthreads();
        // B^T M^-1 F - G
        for (unsigned int row = 0; row < n_patch_dofs_dg; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val =
                    -gpu_data.eigenvalues[patch_type * matrix_size +
                                          matrix_size_U + matrix_size_P +
                                          n_patch_dofs_rt * n_patch_dofs_dg +
                                          row * n_patch_dofs_rt + tid +
                                          i * blockDim.x] *
                    shared_data.tmp[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_src[n_patch_dofs_rt + row], val);
                }
          }
        __syncthreads();
        for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs_dg)
            {
              shared_data.tmp[n_patch_dofs + tid + i * blockDim.x] =
                -shared_data.local_src[n_patch_dofs_rt + tid + i * blockDim.x];
              shared_data
                .tmp[n_patch_dofs + n_patch_dofs_dg + tid + i * blockDim.x] =
                -shared_data.local_src[n_patch_dofs_rt + tid + i * blockDim.x];
            }
        __syncthreads();

        // if (tid == 0)
        //   {
        //     for (unsigned int i = 0; i < n_patch_dofs_dg; ++i)
        //       printf("%.3e ", shared_data.local_src[n_patch_dofs_rt + i]);
        //     printf("\nDEVICE BMF-G \n");
        //   }

        // B^T M^-1 B P = B^T M^-1 F - G
        solver_CG<n_patch_dofs_dg, Number>(
          &gpu_data.eigenvalues[patch_type * matrix_size + matrix_size_U],
          &shared_data.tmp[n_patch_dofs_rt],
          &shared_data.tmp[n_patch_dofs],
          &shared_data.tmp[n_patch_dofs + n_patch_dofs_dg],
          &shared_data.tmp[n_patch_dofs + n_patch_dofs_dg * 2]);
        __syncthreads();

        if (tid == 0)
          {
            for (unsigned int i = 0; i < n_patch_dofs_dg; ++i)
              printf("%.3e ",
                     shared_data.tmp[n_patch_dofs + 3 * n_patch_dofs_dg + i]);
            printf("\nDEVICE P \n");
          }

        for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs_dg)
            shared_data.local_dst[n_patch_dofs_rt + tid + i * blockDim.x] +=
              shared_data
                .tmp[n_patch_dofs + 3 * n_patch_dofs_dg + tid + i * blockDim.x];

        /// M U = F - B P
        // F - B P
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_dg / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_dg)
                {
                  auto val =
                    -gpu_data
                       .eigenvalues[patch_type * matrix_size + matrix_size_U +
                                    matrix_size_P + row * n_patch_dofs_dg +
                                    tid + i * blockDim.x] *
                    shared_data.tmp[n_patch_dofs + 3 * n_patch_dofs_dg + tid +
                                    i * blockDim.x];

                  atomicAdd(&shared_data.local_src[row], val);
                }
          }
        __syncthreads();
        // M U = F - B P
        for (unsigned int row = 0; row < n_patch_dofs_rt; ++row)
          {
            for (unsigned int i = 0; i < n_patch_dofs_rt / blockDim.x + 1; ++i)
              if (tid + i * blockDim.x < n_patch_dofs_rt)
                {
                  auto val = gpu_data.eigenvalues[patch_type * matrix_size +
                                                  row * n_patch_dofs_rt + tid +
                                                  i * blockDim.x] *
                             shared_data.local_src[tid + i * blockDim.x];

                  atomicAdd(&shared_data.local_dst[row], val);
                }
          }
        __syncthreads();

        // store
        for (unsigned int i = 0; i < n_patch_dofs / blockDim.x + 1; ++i)
          if (tid + i * blockDim.x < n_patch_dofs)
            {
              unsigned int global_dof_index =
                gpu_data.patch_dof_smooth[patch * n_patch_dofs_all +
                                          h_interior[tid + i * blockDim.x]];

              dst[global_dof_index] =
                shared_data.local_dst[tid + i * blockDim.x] *
                gpu_data.relaxation;
            }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver>
  __global__
    typename std::enable_if<local_solver != LocalSolverVariant::Direct &&
                            local_solver != LocalSolverVariant::SchurDirect &&
                            local_solver != LocalSolverVariant::SchurIter &&
                            local_solver !=
                              LocalSolverVariant::SchurTensorProduct &&
                            local_solver != LocalSolverVariant::Uzawa>::type
    loop_kernel_global(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    printf("Not Impl.!!!\n");
  }


  // // TODO: Exact solver
  // template <int dim,
  //           int fe_degree,
  //           typename Number,
  //           LaplaceVariant     lapalace,
  //           LocalSolverVariant local_solver>
  // __global__ void
  // loop_kernel_fused_cf(
  //   const Number                                                 *src,
  //   Number                                                       *dst,
  //   const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  // {
  //   constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 1;
  //   constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
  //   constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
  //   constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

  //   const unsigned int n_dofs_per_dim  = gpu_data.n_dofs_per_dim;
  //   const unsigned int patch_per_block = gpu_data.patch_per_block;
  //   const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
  //   const unsigned int patch       = local_patch + patch_per_block *
  //   blockIdx.x; const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

  //   SharedDataSmoother<dim, Number, SmootherVariant::ConflictFree,
  //   local_solver>
  //     shared_data(get_shared_data_ptr<Number>(),
  //                 patch_per_block,
  //                 n_dofs_1d,
  //                 local_dim);

  //   if (patch < gpu_data.n_patches)
  //     {
  //       shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //       shared_data.local_laplace[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];
  //       shared_data.local_bilaplace[threadIdx.y * n_dofs_1d + local_tid_x] =
  //         gpu_data.smooth_bilaplace_1d[threadIdx.y * n_dofs_1d +
  //         local_tid_x];

  //       for (unsigned int z = 0; z < n_dofs_z; ++z)
  //         {
  //           const unsigned int index = local_patch * local_dim +
  //                                      z * n_dofs_1d * n_dofs_1d +
  //                                      threadIdx.y * n_dofs_1d + local_tid_x;

  //           const unsigned int global_dof_indices =
  //             z * n_dofs_per_dim * n_dofs_per_dim +
  //             threadIdx.y * n_dofs_per_dim + local_tid_x +
  //             gpu_data.first_dof[patch];

  //           shared_data.local_src[index] = src[global_dof_indices];

  //           shared_data.local_dst[index] = dst[global_dof_indices];
  //         }

  //       // evaluate_smooth_cf<dim, fe_degree, Number, lapalace,
  //       local_solver>(
  //       //   local_patch, &shared_data, &gpu_data);

  //       if (dim == 2)
  //         {
  //           unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

  //           if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //             {
  //               int row = linear_tid / (n_dofs_1d - 2) + 1;
  //               int col = linear_tid % (n_dofs_1d - 2) + 1;

  //               const unsigned int index = 2 * local_patch * local_dim +
  //                                          (row - 1) * (n_dofs_1d - 2) + col
  //                                          - 1;

  //               const unsigned int global_dof_indices =
  //                 row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

  //               dst[global_dof_indices] =
  //                 shared_data.tmp[index] * gpu_data.relaxation;
  //             }
  //         }
  //       else if (dim == 3)
  //         {
  //           for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
  //             {
  //               unsigned int linear_tid = local_tid_x + threadIdx.y *
  //               n_dofs_1d;

  //               if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
  //                 {
  //                   unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
  //                   unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

  //                   unsigned int index = 2 * local_patch * local_dim +
  //                                        z * n_dofs_1d * n_dofs_1d +
  //                                        (row - 1) * (n_dofs_1d - 2) + col -
  //                                        1;

  //                   const unsigned int global_dof_indices =
  //                     z * n_dofs_per_dim * n_dofs_per_dim +
  //                     row * n_dofs_per_dim + col + gpu_data.first_dof[patch];

  //                   dst[global_dof_indices] =
  //                     shared_data.tmp[index] * gpu_data.relaxation;
  //                 }
  //             }
  //         }
  //     }
  // }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH
