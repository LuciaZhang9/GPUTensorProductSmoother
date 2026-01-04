/**
 * @file cuda_mg_transfer.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the grid transfer operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MG_TRANSFER_TEMPLATE_CUH
#define MG_TRANSFER_TEMPLATE_CUH

#include <deal.II/grid/grid_generator.h>

#include "cuda_mg_transfer.cuh"
#include "cuda_vector.cuh"
#include "transfer_internal.h"

namespace PSMF
{

  enum TransferVariant
  {
    PROLONGATION,
    RESTRICTION
  };

  template <int dim, int fe_degree, typename Number>
  class MGTransferHelper
  {
  protected:
    static constexpr unsigned int n_coarse =
      dim * Util::pow(fe_degree + 1, dim - 1) * (fe_degree + 2) +
      Util::pow(fe_degree + 1, dim);

    static constexpr unsigned int n_fine =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3) +
      Util::pow(2 * fe_degree + 2, dim);

    Number             *values;
    const Number       *weights;
    const Number       *shape_values;
    const unsigned int *row_ptr;
    const unsigned int *col_idx;
    const unsigned int *dof_indices_coarse;
    const unsigned int *dof_indices_fine;

    __device__
    MGTransferHelper(Number             *buf,
                     const Number       *w,
                     const Number       *shvals,
                     const unsigned int *row_ptr_,
                     const unsigned int *col_idx_,
                     const unsigned int *idx_coarse,
                     const unsigned int *idx_fine)
      : values(buf)
      , weights(w)
      , shape_values(shvals)
      , row_ptr(row_ptr_)
      , col_idx(col_idx_)
      , dof_indices_coarse(idx_coarse)
      , dof_indices_fine(idx_fine)
    {}

    template <int kernel>
    __device__ void
    reduce()
    {}


    __device__ void
    weigh_values()
    {
      for (int i = 0; i < n_fine / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_fine)
          values[threadIdx.x + i * Util::BLOCK_DIM] *=
            weights[threadIdx.x + i * Util::BLOCK_DIM];
    }
  };

  template <int dim, int fe_degree, typename Number>
  class MGProlongateHelper : public MGTransferHelper<dim, fe_degree, Number>
  {
    using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::n_fine;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
    using MGTransferHelper<dim, fe_degree, Number>::values;
    using MGTransferHelper<dim, fe_degree, Number>::shape_values;
    using MGTransferHelper<dim, fe_degree, Number>::row_ptr;
    using MGTransferHelper<dim, fe_degree, Number>::col_idx;
    using MGTransferHelper<dim, fe_degree, Number>::weights;

  public:
    static constexpr TransferVariant transfer_variant = PROLONGATION;

    __device__
    MGProlongateHelper(Number             *buf,
                       const Number       *w,
                       const Number       *shvals,
                       const unsigned int *row_ptr,
                       const unsigned int *col_idx,
                       const unsigned int *idx_coarse,
                       const unsigned int *idx_fine)
      : MGTransferHelper<dim, fe_degree, Number>(buf,
                                                 w,
                                                 shvals,
                                                 row_ptr,
                                                 col_idx,
                                                 idx_coarse,
                                                 idx_fine)
    {}

    __device__ void
    run(Number *dst, const Number *src)
    {
      read_coarse(src);
      __syncthreads();

      reduce_csr();
      __syncthreads();

      this->weigh_values();
      __syncthreads();

      write_fine(dst);
    }

  private:
    __device__ void
    read_coarse(const Number *vec)
    {
      for (int i = 0; i < n_coarse / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_coarse)
          values[threadIdx.x + i * Util::BLOCK_DIM] =
            vec[dof_indices_coarse[threadIdx.x + i * Util::BLOCK_DIM]];
    }

    __device__ void
    reduce_csr()
    {
      Number sum[n_fine / Util::BLOCK_DIM + 1];

      for (int i = 0; i < n_fine / Util::BLOCK_DIM + 1; ++i)
        {
          sum[i] = 0;
          if (threadIdx.x + i * Util::BLOCK_DIM < n_fine)
            for (auto j = row_ptr[threadIdx.x + i * Util::BLOCK_DIM];
                 j < row_ptr[threadIdx.x + i * Util::BLOCK_DIM + 1];
                 ++j)
              sum[i] += shape_values[j] * values[col_idx[j]];
        }
      __syncthreads();
      for (int i = 0; i < n_fine / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_fine)
          values[threadIdx.x + i * Util::BLOCK_DIM] = sum[i];
    }

    __device__ void
    write_fine(Number *vec) const
    {
      for (int i = 0; i < n_fine / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_fine)
          atomicAdd(&vec[dof_indices_fine[threadIdx.x + i * Util::BLOCK_DIM]],
                    values[threadIdx.x + i * Util::BLOCK_DIM]);
    }
  };

  template <int dim, int fe_degree, typename Number>
  class MGRestrictHelper : public MGTransferHelper<dim, fe_degree, Number>
  {
    using MGTransferHelper<dim, fe_degree, Number>::n_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::n_fine;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_coarse;
    using MGTransferHelper<dim, fe_degree, Number>::dof_indices_fine;
    using MGTransferHelper<dim, fe_degree, Number>::values;
    using MGTransferHelper<dim, fe_degree, Number>::shape_values;
    using MGTransferHelper<dim, fe_degree, Number>::row_ptr;
    using MGTransferHelper<dim, fe_degree, Number>::col_idx;
    using MGTransferHelper<dim, fe_degree, Number>::weights;

  public:
    static constexpr TransferVariant transfer_variant = RESTRICTION;

    __device__
    MGRestrictHelper(Number             *buf,
                     const Number       *w,
                     const Number       *shvals,
                     const unsigned int *row_ptr,
                     const unsigned int *col_idx,
                     const unsigned int *idx_coarse,
                     const unsigned int *idx_fine)
      : MGTransferHelper<dim, fe_degree, Number>(buf,
                                                 w,
                                                 shvals,
                                                 row_ptr,
                                                 col_idx,
                                                 idx_coarse,
                                                 idx_fine)
    {}

    __device__ void
    run(Number *dst, const Number *src)
    {
      read_fine(src);
      __syncthreads();

      this->weigh_values();
      __syncthreads();

      reduce_csr();
      __syncthreads();

      write_coarse(dst);
    }

  private:
    __device__ void
    read_fine(const Number *vec)
    {
      for (int i = 0; i < n_fine / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_fine)
          values[threadIdx.x + i * Util::BLOCK_DIM] =
            vec[dof_indices_fine[threadIdx.x + i * Util::BLOCK_DIM]];
    }

    __device__ void
    reduce_csr()
    {
      Number sum[n_coarse / Util::BLOCK_DIM + 1];

      for (int i = 0; i < n_coarse / Util::BLOCK_DIM + 1; ++i)
        {
          sum[i] = 0;
          if (threadIdx.x + i * Util::BLOCK_DIM < n_coarse)
            for (auto j = row_ptr[threadIdx.x + i * Util::BLOCK_DIM];
                 j < row_ptr[threadIdx.x + i * Util::BLOCK_DIM + 1];
                 ++j)
              sum[i] += shape_values[j] * values[col_idx[j]];
        }

      __syncthreads();
      for (int i = 0; i < n_coarse / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_coarse)
          values[threadIdx.x + i * Util::BLOCK_DIM] = sum[i];
    }

    __device__ void
    write_coarse(Number *vec) const
    {
      for (int i = 0; i < n_coarse / Util::BLOCK_DIM + 1; ++i)
        if (threadIdx.x + i * Util::BLOCK_DIM < n_coarse)
          atomicAdd(&vec[dof_indices_coarse[threadIdx.x + i * Util::BLOCK_DIM]],
                    values[threadIdx.x + i * Util::BLOCK_DIM]);
    }
  };


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


  template <int dim, int degree, typename loop_body, typename Number>
  __global__ void
  mg_kernel(Number             *dst,
            const Number       *src,
            const Number       *weights,
            const Number       *shape_values,
            const unsigned int *row_ptr,
            const unsigned int *col_idx,
            const unsigned int *dof_indices_coarse,
            const unsigned int *dof_indices_fine,
            const unsigned int  n_child_cell_dofs)
  {
    constexpr unsigned int n_coarse =
      dim * Util::pow(degree + 1, dim - 1) * (degree + 2) +
      Util::pow(degree + 1, dim);

    const unsigned int coarse_cell = blockIdx.x;

    loop_body body(get_shared_data_ptr<Number>(),
                   weights + coarse_cell * n_child_cell_dofs,
                   shape_values,
                   row_ptr,
                   col_idx,
                   dof_indices_coarse + coarse_cell * n_coarse,
                   dof_indices_fine + coarse_cell * n_child_cell_dofs);

    body.run(dst, src);
  }



  template <int dim, typename Number>
  template <template <int, int, typename> class loop_body, int degree>
  void
  MGTransferCUDA<dim, Number>::coarse_cell_loop(
    const unsigned int                                             fine_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    constexpr unsigned int n_coarse =
      dim * Util::pow(degree + 1, dim - 1) * (degree + 2) +
      Util::pow(degree + 1, dim);
    constexpr unsigned int n_fine_dofs =
      dim * Util::pow(2 * degree + 2, dim - 1) * (2 * degree + 3) +
      Util::pow(2 * degree + 2, dim);

    constexpr unsigned int n_fine_size = n_fine_dofs * sizeof(Number);

    AssertDimension(n_child_cell_dofs, n_fine_dofs);

    constexpr TransferVariant transfer_vatiant =
      loop_body<dim, degree, Number>::transfer_variant;

    const unsigned int n_coarse_cells = n_owned_level_cells[fine_level - 1];

    // kernel parameters
    dim3 bk_dim(Util::BLOCK_DIM, 1, 1);
    dim3 gd_dim(n_coarse_cells);

    AssertCuda(cudaFuncSetAttribute(
      mg_kernel<dim, degree, loop_body<dim, degree, Number>, Number>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      n_fine_size));

    mg_kernel<dim, degree, loop_body<dim, degree, Number>>
      <<<gd_dim, bk_dim, n_fine_size>>>(
        dst.get_values(),
        src.get_values(),
        weights_on_refined[fine_level - 1].get_values(),
        transfer_matrix_val[transfer_vatiant].get_values(),
        transfer_matrix_row_ptr[transfer_vatiant].get_values(),
        transfer_matrix_col_idx[transfer_vatiant].get_values(),
        level_dof_indices_parent[fine_level - 1].get_values(),
        level_dof_indices_child[fine_level].get_values(),
        n_child_cell_dofs);

    AssertCudaKernel();
  }

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number>::MGTransferCUDA()
    : fe_degree(0)
    , element_is_continuous(false)
    , n_components(0)
    , n_child_cell_dofs(0)
  {}

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number>::MGTransferCUDA(const MGConstrainedDoFs &mg_c)
    : fe_degree(0)
    , element_is_continuous(false)
    , n_components(0)
    , n_child_cell_dofs(0)
  {
    this->mg_constrained_dofs = &mg_c;
  }

  template <int dim, typename Number>
  MGTransferCUDA<dim, Number>::~MGTransferCUDA()
  {}

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::initialize_constraints(
    const MGConstrainedDoFs &mg_c)
  {
    this->mg_constrained_dofs = &mg_c;
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::clear()
  {
    fe_degree             = 0;
    element_is_continuous = false;
    n_components          = 0;
    n_child_cell_dofs     = 0;
    level_dof_indices.clear();
    child_offset_in_parent.clear();
    n_owned_level_cells.clear();
    weights_on_refined.clear();
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::build(
    const DoFHandler<dim, dim> &mg_dof_velocity,
    const DoFHandler<dim, dim> &mg_dof_pressure,
    const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      &external_partitioners)
  {
    Assert(mg_dof_velocity.has_level_dofs() && mg_dof_pressure.has_level_dofs(),
           ExcMessage(
             "The underlying DoFHandler object has not had its "
             "distribute_mg_dofs() function called, but this is a prerequisite "
             "for multigrid transfers. You will need to call this function, "
             "probably close to where you already call distribute_dofs()."));

    fill_copy_indices(mg_dof_velocity);

    const unsigned int n_levels =
      mg_dof_velocity.get_triangulation().n_global_levels();

    std::vector<std::vector<Number>>       weights_host;
    std::vector<std::vector<unsigned int>> level_dof_indices_parent_host;
    std::vector<std::vector<unsigned int>> level_dof_indices_child_host;
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
      parent_child_connect;

    std::vector<Table<2, unsigned int>> copy_indices_global_mine;
    MGLevelObject<LinearAlgebra::distributed::Vector<Number>>
      ghosted_level_vector;
    std::vector<std::vector<std::vector<unsigned short>>>
      dirichlet_indices_host;

    std::vector<internal::CSRMatrix<Number>> transfer_matrix;

    ghosted_level_vector.resize(0, n_levels - 1);

    vector_partitioners.resize(0, n_levels - 1);
    for (unsigned int level = 0; level <= ghosted_level_vector.max_level();
         ++level)
      vector_partitioners[level] =
        ghosted_level_vector[level].get_partitioner();


    internal::ElementInfo<Number> elem_info;
    internal::setup_transfer<dim, Number>(mg_dof_velocity,
                                          mg_dof_pressure,
                                          this->mg_constrained_dofs,
                                          external_partitioners,
                                          elem_info,
                                          level_dof_indices_parent_host,
                                          level_dof_indices_child_host,
                                          n_owned_level_cells,
                                          weights_host,
                                          copy_indices_global_mine,
                                          vector_partitioners);

    // unpack element info data
    fe_degree             = elem_info.fe_degree;
    element_is_continuous = elem_info.element_is_continuous;
    n_components          = elem_info.n_components;
    n_child_cell_dofs     = elem_info.n_child_cell_dofs;

    //---------------------------------------------------------------------------
    // transfer stuff from host to device
    //---------------------------------------------------------------------------
    setup_prolongatino_matrix(mg_dof_velocity,
                              mg_dof_pressure,
                              transfer_matrix);
    transfer_matrix_val.resize(2);
    transfer_matrix_row_ptr.resize(2);
    transfer_matrix_col_idx.resize(2);
    for (unsigned int i = 0; i < transfer_matrix.size(); ++i)
      {
        copy_to_device(transfer_matrix_val[i], transfer_matrix[i].values);
        copy_to_device(transfer_matrix_row_ptr[i], transfer_matrix[i].row_ptr);
        copy_to_device(transfer_matrix_col_idx[i], transfer_matrix[i].col_idx);
      }

    level_dof_indices.resize(n_levels);
    level_dof_indices_parent.resize(n_levels);
    level_dof_indices_child.resize(n_levels);

    for (unsigned int l = 0; l < n_levels; l++)
      {
        copy_to_device(level_dof_indices_parent[l],
                       level_dof_indices_parent_host[l]);
        copy_to_device(level_dof_indices_child[l],
                       level_dof_indices_child_host[l]);

        // std::cout << "Level " << l << std::endl;
        // for (auto ind : level_dof_indices_parent_host[l])
        //   std::cout << ind << " ";
        // std::cout << "\n\n";
      }

    weights_on_refined.resize(n_levels - 1);
    for (unsigned int l = 0; l < n_levels - 1; l++)
      {
        copy_to_device(weights_on_refined[l], weights_host[l]);

        // std::cout << "Level " << l << std::endl;
        // for (auto ind : weights_host[l])
        //   std::cout << ind << " ";
        // std::cout << "\n\n";
      }

    child_offset_in_parent.resize(n_levels - 1);

    std::vector<types::global_dof_index> dirichlet_index_vector;
    dirichlet_indices.resize(n_levels);
    if (this->mg_constrained_dofs != nullptr &&
        mg_constrained_dofs->have_boundary_indices())
      {
        for (unsigned int l = 0; l < n_levels; l++)
          {
            mg_constrained_dofs->get_boundary_indices(l).fill_index_vector(
              dirichlet_index_vector);
            copy_to_device(dirichlet_indices[l], dirichlet_index_vector);
          }
      }
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::prolongate(
    const unsigned int                                             to_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    dst = 0;
    prolongate_and_add(to_level, dst, src);
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::prolongate_and_add(
    const unsigned int                                             to_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    Assert((to_level >= 1) && (to_level <= level_dof_indices.size()),
           ExcIndexRange(to_level, 1, level_dof_indices.size() + 1));

    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> src_with_bc(
      src);
    set_mg_constrained_dofs(src_with_bc, to_level - 1, 0);

    if (fe_degree == 1)
      coarse_cell_loop<MGProlongateHelper, 1>(to_level, dst, src_with_bc);
    else if (fe_degree == 2)
      coarse_cell_loop<MGProlongateHelper, 2>(to_level, dst, src_with_bc);
    else if (fe_degree == 3)
      coarse_cell_loop<MGProlongateHelper, 3>(to_level, dst, src_with_bc);
    else if (fe_degree == 4)
      coarse_cell_loop<MGProlongateHelper, 4>(to_level, dst, src_with_bc);
    else if (fe_degree == 5)
      coarse_cell_loop<MGProlongateHelper, 5>(to_level, dst, src_with_bc);
    else if (fe_degree == 6)
      coarse_cell_loop<MGProlongateHelper, 6>(to_level, dst, src_with_bc);
    else if (fe_degree == 7)
      coarse_cell_loop<MGProlongateHelper, 7>(to_level, dst, src_with_bc);
    else if (fe_degree == 8)
      coarse_cell_loop<MGProlongateHelper, 8>(to_level, dst, src_with_bc);
    else if (fe_degree == 9)
      coarse_cell_loop<MGProlongateHelper, 9>(to_level, dst, src_with_bc);
    else if (fe_degree == 10)
      coarse_cell_loop<MGProlongateHelper, 10>(to_level, dst, src_with_bc);
    else
      AssertThrow(false,
                  ExcNotImplemented("Only degrees 1 through 10 implemented."));
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::restrict_and_add(
    const unsigned int                                             from_level,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
    const
  {
    Assert((from_level >= 1) && (from_level <= level_dof_indices.size()),
           ExcIndexRange(from_level, 1, level_dof_indices.size() + 1));

    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> increment;
    increment.reinit(dst,
                     false); // resize to correct size and initialize to 0

    if (fe_degree == 1)
      coarse_cell_loop<MGRestrictHelper, 1>(from_level, increment, src);
    else if (fe_degree == 2)
      coarse_cell_loop<MGRestrictHelper, 2>(from_level, increment, src);
    else if (fe_degree == 3)
      coarse_cell_loop<MGRestrictHelper, 3>(from_level, increment, src);
    else if (fe_degree == 4)
      coarse_cell_loop<MGRestrictHelper, 4>(from_level, increment, src);
    else if (fe_degree == 5)
      coarse_cell_loop<MGRestrictHelper, 5>(from_level, increment, src);
    else if (fe_degree == 6)
      coarse_cell_loop<MGRestrictHelper, 6>(from_level, increment, src);
    else if (fe_degree == 7)
      coarse_cell_loop<MGRestrictHelper, 7>(from_level, increment, src);
    else if (fe_degree == 8)
      coarse_cell_loop<MGRestrictHelper, 8>(from_level, increment, src);
    else if (fe_degree == 9)
      coarse_cell_loop<MGRestrictHelper, 9>(from_level, increment, src);
    else if (fe_degree == 10)
      coarse_cell_loop<MGRestrictHelper, 10>(from_level, increment, src);
    else
      AssertThrow(false,
                  ExcNotImplemented("Only degrees 1 through 10 implemented."));

    set_mg_constrained_dofs(increment, from_level - 1, 0);

    dst.add(1., increment);
  }

  template <typename Number>
  __global__ void
  set_mg_constrained_dofs_kernel(Number             *vec,
                                 const unsigned int *indices,
                                 unsigned int        len,
                                 Number              val)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
      {
        vec[indices[idx]] = val;
      }
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::set_mg_constrained_dofs(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec,
    unsigned int                                                   level,
    Number                                                         val) const
  {
    const unsigned int len = dirichlet_indices[level].size();
    if (len > 0)
      {
        const unsigned int bksize  = 256;
        const unsigned int nblocks = (len - 1) / bksize + 1;
        dim3               bk_dim(bksize);
        dim3               gd_dim(nblocks);

        set_mg_constrained_dofs_kernel<<<gd_dim, bk_dim>>>(
          vec.get_values(), dirichlet_indices[level].get_values(), len, val);
        AssertCudaKernel();
      }
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number>::copy_to_mg(
    const DoFHandler<dim, spacedim> &mg_dof,
    MGLevelObject<LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
                                                                         &dst,
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src)
    const
  {
    AssertIndexRange(dst.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(dst.min_level(), dst.max_level() + 1);

    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      {
        dst[level].reinit(mg_dof.n_dofs(level));
      }

    if (perform_plain_copy)
      {
        // if the finest multigrid level covers the whole domain (i.e., no
        // adaptive refinement) and the numbering of the finest level DoFs and
        // the global DoFs are the same, we can do a plain copy

        AssertDimension(dst[dst.max_level()].size(), src.size());

        plain_copy<false>(dst[dst.max_level()], src);

        return;
      }

    for (unsigned int level = dst.max_level() + 1; level != dst.min_level();)
      {
        --level;
        auto &dst_level = dst[level];

        copy_with_indices(dst_level,
                          src,
                          copy_indices[level].level_indices,
                          copy_indices[level].global_indices);
      }
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number>::copy_from_mg(
    const DoFHandler<dim, spacedim>                                &mg_dof,
    LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
    const MGLevelObject<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src) const
  {
    AssertIndexRange(src.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(src.min_level(), src.max_level() + 1);

    (void)mg_dof;

    if (perform_plain_copy)
      {
        AssertDimension(dst.size(), src[src.max_level()].size());
        plain_copy<false>(dst, src[src.max_level()]);
        return;
      }

    dst = 0;
    for (unsigned int level = src.min_level(); level <= src.max_level();
         ++level)
      {
        const auto &src_level = src[level];

        copy_with_indices(dst,
                          src_level,
                          copy_indices[level].global_indices,
                          copy_indices[level].level_indices);
      }
  }

  template <int dim, typename Number>
  template <int spacedim, typename Number2>
  void
  MGTransferCUDA<dim, Number>::copy_from_mg_add(
    const DoFHandler<dim, spacedim>                                &mg_dof,
    LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
    const MGLevelObject<
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src) const
  {
    AssertIndexRange(src.max_level(),
                     mg_dof.get_triangulation().n_global_levels());
    AssertIndexRange(src.min_level(), src.max_level() + 1);

    (void)mg_dof;

    if (perform_plain_copy)
      {
        AssertDimension(dst.size(), src[src.max_level()].size());
        plain_copy<true>(dst, src[src.max_level()]);
        return;
      }

    auto temp = dst;
    for (unsigned int level = src.min_level(); level <= src.max_level();
         ++level)
      {
        const auto &src_level = src[level];

        copy_with_indices(temp,
                          src_level,
                          copy_indices[level].global_indices,
                          copy_indices[level].level_indices);
      }
    dst += temp;
  }

  template <int dim, typename Number>
  std::size_t
  MGTransferCUDA<dim, Number>::memory_consumption() const
  {
    std::size_t memory = 0;
    memory += MemoryConsumption::memory_consumption(copy_indices);
    memory += MemoryConsumption::memory_consumption(level_dof_indices);
    memory += MemoryConsumption::memory_consumption(child_offset_in_parent);
    memory += MemoryConsumption::memory_consumption(n_owned_level_cells);
    memory += prolongation_matrix_1d.memory_consumption();
    memory += MemoryConsumption::memory_consumption(weights_on_refined);
    memory += MemoryConsumption::memory_consumption(dirichlet_indices);
    return memory;
  }

  template <int dim, typename Number>
  template <typename VectorType, typename VectorType2>
  void
  MGTransferCUDA<dim, Number>::copy_to_device(VectorType        &device,
                                              const VectorType2 &host)
  {
    LinearAlgebra::ReadWriteVector<typename VectorType::value_type> rw_vector(
      host.size());
    device.reinit(host.size());
    for (unsigned int i = 0; i < host.size(); ++i)
      rw_vector[i] = host[i];
    device.import(rw_vector, VectorOperation::insert);
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::setup_prolongatino_matrix(
    const DoFHandler<dim, dim>               &mg_dof_velocity,
    const DoFHandler<dim, dim>               &mg_dof_pressure,
    std::vector<internal::CSRMatrix<Number>> &transfer_matrix)
  {
    Triangulation<dim> tr(
      Triangulation<dim>::limit_level_difference_at_vertices);
    GridGenerator::hyper_cube(tr, 0, 1);
    tr.refine_global(1);

    // COO format to CSR format
    std::vector<unsigned int> row, col;
    std::vector<Number>       val;

    // velocity

    DoFHandler<dim> mgdof_v(tr);
    mgdof_v.distribute_dofs(mg_dof_velocity.get_fe());
    mgdof_v.distribute_mg_dofs();
    {
      // MGConstrainedDoFs mg_constrained_dofs;
      // mg_constrained_dofs.initialize(mgdof_v);
      // mg_constrained_dofs.make_zero_boundary_constraints(mgdof_v, {0});

      MGTransferPrebuilt<Vector<Number>> transfer_ref;
      transfer_ref.build(mgdof_v);

      std::string temp_str;

      std::ostringstream oss;
      transfer_ref.print_matrices(oss);

      // transfer_ref.print_matrices(std::cout);

      std::string        data = oss.str();
      std::istringstream iss(data);

      for (std::string line; std::getline(iss, line);)
        {
          std::stringstream str_strm;
          str_strm << line;

          str_strm >> temp_str; // take words into temp_str one by one

          int p1 = temp_str.find("(");
          int p2 = temp_str.find(",");
          int p3 = temp_str.find(")");

          if (p1 < 0)
            continue;
          row.push_back(std::stoi(temp_str.substr(p1 + 1, p2 - p1 - 1)));
          col.push_back(std::stoi(temp_str.substr(p2 + 1, p3 - p2 - 1)));

          str_strm >> temp_str;

          val.push_back(std::stod(temp_str));

          temp_str = ""; // clear temp string
        }
    }

    // pressure

    DoFHandler<dim> mgdof_p(tr);
    mgdof_p.distribute_dofs(mg_dof_pressure.get_fe());
    mgdof_p.distribute_mg_dofs();
    {
      MGTransferPrebuilt<Vector<Number>> transfer_ref;
      transfer_ref.build(mgdof_p);

      std::string temp_str;

      std::ostringstream oss;
      transfer_ref.print_matrices(oss);

      // transfer_ref.print_matrices(std::cout);

      std::string        data = oss.str();
      std::istringstream iss(data);

      for (std::string line; std::getline(iss, line);)
        {
          std::stringstream str_strm;
          str_strm << line;

          str_strm >> temp_str; // take words into temp_str one by one

          int p1 = temp_str.find("(");
          int p2 = temp_str.find(",");
          int p3 = temp_str.find(")");

          if (p1 < 0)
            continue;
          row.push_back(mgdof_v.n_dofs(1) +
                        std::stoi(temp_str.substr(p1 + 1, p2 - p1 - 1)));
          col.push_back(mgdof_v.n_dofs(0) +
                        std::stoi(temp_str.substr(p2 + 1, p3 - p2 - 1)));

          str_strm >> temp_str;

          val.push_back(std::stod(temp_str));

          temp_str = ""; // clear temp string
        }
    }

    std::vector<internal::COOEntry<Number>> coo_entries(row.size());
    for (unsigned int i = 0; i < val.size(); ++i)
      {
        coo_entries[i].row   = row[i];
        coo_entries[i].col   = col[i];
        coo_entries[i].value = val[i];

        // std::cout << row[i] << ", " << col[i] << " " << val[i] << std::endl;
      }

    auto coo_to_csr = [&](auto coo, unsigned int num_rows, unsigned int) {
      // Sort the COO entries by row index
      std::vector<internal::COOEntry<Number>> sorted_coo_entries = coo;
      std::sort(sorted_coo_entries.begin(),
                sorted_coo_entries.end(),
                [](const internal::COOEntry<Number> &a,
                   const internal::COOEntry<Number> &b) {
                  if (a.row == b.row)
                    return a.col < b.col;
                  else
                    return a.row < b.row;
                });

      internal::CSRMatrix<Number> csr_matrix;
      csr_matrix.row_ptr.resize(num_rows + 1);
      csr_matrix.col_idx.reserve(coo.size());
      csr_matrix.values.reserve(coo.size());

      // Initialize row_ptr to all zeros
      std::fill(csr_matrix.row_ptr.begin(), csr_matrix.row_ptr.end(), 0);

      // Count the number of non-zero entries in each row
      for (const auto &entry : sorted_coo_entries)
        {
          csr_matrix.row_ptr[entry.row + 1]++;
        }

      // Cumulative sum of row counts to get row_ptr
      for (unsigned int i = 0; i < num_rows; i++)
        {
          csr_matrix.row_ptr[i + 1] += csr_matrix.row_ptr[i];
        }

      // Copy data from sorted_coo_entries into CSR format
      for (const auto &entry : sorted_coo_entries)
        {
          csr_matrix.col_idx.push_back(entry.col);
          csr_matrix.values.push_back(entry.value);
        }

      return csr_matrix;
    };

    auto prolongation_matrix =
      coo_to_csr(coo_entries,
                 mgdof_v.n_dofs(1) + mgdof_p.n_dofs(1),
                 mgdof_v.n_dofs(0) + mgdof_p.n_dofs(0));
    transfer_matrix.push_back(prolongation_matrix);

    // transpose_coo
    for (unsigned int i = 0; i < coo_entries.size(); ++i)
      std::swap(coo_entries[i].row, coo_entries[i].col);

    auto restriction_matrix = coo_to_csr(coo_entries,
                                         mgdof_v.n_dofs(0) + mgdof_p.n_dofs(0),
                                         mgdof_v.n_dofs(1) + mgdof_p.n_dofs(1));
    transfer_matrix.push_back(restriction_matrix);
  }

  template <int dim, typename Number>
  void
  MGTransferCUDA<dim, Number>::fill_copy_indices(const DoFHandler<dim> &mg_dof)
  {
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_global_mine;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_level_mine;

    dealii::internal::MGTransfer::fill_copy_indices(mg_dof,
                                                    mg_constrained_dofs,
                                                    my_copy_indices,
                                                    my_copy_indices_global_mine,
                                                    my_copy_indices_level_mine);

    const unsigned int nlevels = mg_dof.get_triangulation().n_global_levels();

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        Assert((my_copy_indices_global_mine[level].size() == 0) &&
                 (my_copy_indices_level_mine[level].size() == 0),
               ExcMessage("Only implemented for non-distributed case"));
      }

    copy_indices.resize(nlevels);
    for (unsigned int i = 0; i < nlevels; ++i)
      {
        const unsigned int nmappings = my_copy_indices[i].size();
        std::vector<int>   global_indices(nmappings);
        std::vector<int>   level_indices(nmappings);

        for (unsigned int j = 0; j < nmappings; ++j)
          {
            global_indices[j] = my_copy_indices[i][j].first;
            level_indices[j]  = my_copy_indices[i][j].second;
          }

        copy_to_device(copy_indices[i].global_indices, global_indices);
        copy_to_device(copy_indices[i].level_indices, level_indices);
      }

    // check if we can run a plain copy operation between the global DoFs and
    // the finest level.
    perform_plain_copy =
      (my_copy_indices.back().size() ==
       mg_dof.locally_owned_dofs().n_elements()) &&
      (mg_dof.locally_owned_dofs().n_elements() ==
       mg_dof.locally_owned_mg_dofs(nlevels - 1).n_elements());

    if (perform_plain_copy)
      {
        AssertDimension(my_copy_indices_global_mine.back().size(), 0);
        AssertDimension(my_copy_indices_level_mine.back().size(), 0);

        // check whether there is a renumbering of degrees of freedom on
        // either the finest level or the global dofs, which means that we
        // cannot apply a plain copy
        for (unsigned int i = 0; i < my_copy_indices.back().size(); ++i)
          if (my_copy_indices.back()[i].first !=
              my_copy_indices.back()[i].second)
            {
              perform_plain_copy = false;
              break;
            }
      }
    perform_plain_copy = true;
  }
} // namespace PSMF

#endif // MG_TRANSFER_TEMPLATE_CUH