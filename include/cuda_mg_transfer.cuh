/**
 * @file cuda_mg_transfer.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the grid transfer operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MG_TRANSFER_CUH
#define MG_TRANSFER_CUH

#include <deal.II/base/config.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_internal.h>

#include "cuda_vector.cuh"
#include "transfer_internal.h"
#include "utilities.cuh"

using namespace dealii;

namespace PSMF
{

  struct IndexMapping
  {
    CudaVector<unsigned int> global_indices;
    CudaVector<unsigned int> level_indices;

    std::size_t
    memory_consumption() const
    {
      return global_indices.memory_consumption() +
             level_indices.memory_consumption();
    }
  };


  /**
   * Implementation of the MGTransferBase interface for which the transfer
   * operations is implemented in a matrix-free way based on the interpolation
   * matrices of the underlying finite element. This requires considerably
   * less memory than MGTransferPrebuilt and can also be considerably faster
   * than that variant.
   */
  template <int dim, typename Number>
  class MGTransferCUDA
    : public MGTransferBase<LinearAlgebra::distributed::Vector<
        Number,
        MemorySpace::CUDA>> // public Subscriptor
  {
  public:
    /**
     * Constructor without constraint. Use this constructor only with
     * discontinuous finite elements or with no local refinement.
     */
    MGTransferCUDA();

    /**
     * Constructor with constraints. Equivalent to the default constructor
     * followed by initialize_constraints().
     */
    MGTransferCUDA(const MGConstrainedDoFs &mg_constrained_dofs);

    /**
     * Destructor.
     */
    ~MGTransferCUDA();

    /**
     * Initialize the constraints to be used in build().
     */
    void
    initialize_constraints(const MGConstrainedDoFs &mg_constrained_dofs);

    /**
     * Reset the object to the state it had right after the default
     * constructor.
     */
    void
    clear();

    /**
     * Actually build the information for the prolongation for each level.
     */
    void
    build(
      const DoFHandler<dim, dim> &mg_dof_velocity,
      const DoFHandler<dim, dim> &mg_dof_pressure,
      const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        &external_partitioners =
          std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>());

    /**
     * Prolongate a vector from level <tt>to_level-1</tt> to level
     * <tt>to_level</tt> using the embedding matrices of the underlying finite
     * element. The previous content of <tt>dst</tt> is overwritten.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the coarser level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * finer level.
     */
    void
    prolongate(
      const unsigned int                                             to_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Prolongate a vector from level <tt>to_level-1</tt> to level
     * <tt>to_level</tt> using the embedding matrices of the underlying finite
     * element, summing into the previous content of <tt>dst</tt>.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the coarser level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * finer level.
     */
    void
    prolongate_and_add(
      const unsigned int                                             to_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Restrict a vector from level <tt>from_level</tt> to level
     * <tt>from_level-1</tt> using the transpose operation of the prolongate()
     * method. If the region covered by cells on level <tt>from_level</tt> is
     * smaller than that of level <tt>from_level-1</tt> (local refinement),
     * then some degrees of freedom in <tt>dst</tt> are active and will not be
     * altered. For the other degrees of freedom, the result of the
     * restriction is added.
     *
     * @param src is a vector with as many elements as there are degrees of
     * freedom on the finer level involved.
     *
     * @param dst has as many elements as there are degrees of freedom on the
     * coarser level.
     */
    void
    restrict_and_add(
      const unsigned int                                             from_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    /**
     * Transfer from multi-level vector to normal vector.
     *
     * Copies data from active portions of an MGVector into the respective
     * positions of a <tt>Vector<number></tt>. In order to keep the result
     * consistent, constrained degrees of freedom are set to zero.
     */
    template <int spacedim, typename Number2>
    void
    copy_to_mg(
      const DoFHandler<dim, spacedim> &mg_dof,
      MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>     &dst,
      const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src)
      const;

    /**
     * Transfer from multi-level vector to normal vector.
     *
     * Copies data from active portions of an MGVector into the respective
     * positions of a <tt>Vector<number></tt>. In order to keep the result
     * consistent, constrained degrees of freedom are set to zero.
     */
    template <int spacedim, typename Number2>
    void
    copy_from_mg(
      const DoFHandler<dim, spacedim>                                &mg_dof,
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
      const MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src)
      const;

    /**
     * Add a multi-level vector to a normal vector.
     *
     * Works as the previous function, but probably not for continuous
     * elements.
     */
    template <int spacedim, typename Number2>
    void
    copy_from_mg_add(
      const DoFHandler<dim, spacedim>                                &mg_dof,
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
      const MGLevelObject<
        LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>> &src)
      const;

    /**
     * Memory used by this object.
     */
    std::size_t
    memory_consumption() const;

  private:
    /**
     * A variable storing the degree of the finite element contained in the
     * DoFHandler passed to build(). The selection of the computational kernel
     * is based on this number.
     */
    unsigned int fe_degree;

    /**
     * A variable storing whether the element is continuous and there is a
     * joint degree of freedom in the center of the 1D line.
     */
    bool element_is_continuous;

    /**
     * A variable storing the number of components in the finite element
     * contained in the DoFHandler passed to build().
     */
    unsigned int n_components;

    /**
     * A variable storing the number of degrees of freedom on all child cells.
     * It is <tt>2<sup>dim</sup>*fe.dofs_per_cell</tt> for DG elements and
     * somewhat less for continuous elements.
     */
    unsigned int n_child_cell_dofs;

    /**
     * This variable holds the indices for cells on a given level, extracted
     * from DoFHandler for fast access. All DoF indices on a given level are
     * stored as a plain array (since this class assumes constant DoFs per
     * cell). To index into this array, use the cell number times
     * dofs_per_cell.
     *
     * This array first is arranged such that all locally owned level cells
     * come first (found in the variable n_owned_level_cells) and then other
     * cells necessary for the transfer to the next level.
     *
     */
    std::vector<CudaVector<unsigned int>> level_dof_indices;

    std::vector<CudaVector<unsigned int>> level_dof_indices_parent;
    std::vector<CudaVector<unsigned int>> level_dof_indices_child;

    /**
     * A variable storing the connectivity from parent to child cell numbers
     * for each level.
     */
    std::vector<CudaVector<unsigned int>> child_offset_in_parent;

    /**
     * A variable storing the number of cells owned on a given process (sets
     * the bounds for the worker loops) for each level.
     */
    std::vector<unsigned int> n_owned_level_cells;

    /**
     * Holds the (todo: one-dimensional) embedding (prolongation) matrix from
     * mother element to the children. {P, R}
     */
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
      prolongation_matrix_1d;

    std::vector<LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
      transfer_matrix_val;

    std::vector<CudaVector<unsigned int>> transfer_matrix_row_ptr;
    std::vector<CudaVector<unsigned int>> transfer_matrix_col_idx;

    /**
     * For continuous elements, restriction is not additive and we need to
     * weight the result at the end of prolongation (and at the start of
     * restriction) by the valence of the degrees of freedom, i.e., on how
     * many elements they appear. We store the data in vectorized form to
     * allow for cheap access. Moreover, we utilize the fact that we only need
     * to store <tt>3<sup>dim</sup></tt> indices.
     *
     * Data is organized in terms of each level (outer vector) and the cells
     * on each level (inner vector).
     */
    std::vector<LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>>
      weights_on_refined;

    /**
     * Mapping for the copy_to_mg() and copy_from_mg() functions. Here only
     * index pairs locally owned is stored.
     * The data is organized as follows: one table per level. This table has
     * two rows. The first row contains the global index, the second one the
     * level index.
     */
    std::vector<IndexMapping> copy_indices;

    /**
     * This variable stores whether the copy operation from the global to the
     * level vector is actually a plain copy to the finest level. This means
     * that the grid has no adaptive refinement and the numbering on the
     * finest multigrid level is the same as in the global case.
     */
    bool perform_plain_copy;

    /**
     * A variable storing the local indices of Dirichlet boundary conditions
     * on cells for all levels (outer index), the cells within the levels
     * (second index), and the indices on the cell (inner index).
     */
    std::vector<CudaVector<unsigned int>> dirichlet_indices;

    /**
     * A vector that holds shared pointers to the partitioners of the
     * transfer. These partitioners might be shared with what was passed in
     * from the outside through build() or be shared with the level vectors
     * inherited from MGLevelGlobalTransfer.
     */
    MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
      vector_partitioners;

    /**
     * The mg_constrained_dofs of the level systems.
     */
    SmartPointer<const MGConstrainedDoFs, MGTransferCUDA<dim, Number>>
      mg_constrained_dofs;

    /**
     * Setup the embedding (prolongation) matrix.
     */
    void
    setup_prolongatino_matrix(
      const DoFHandler<dim, dim>               &mg_dof_velocity,
      const DoFHandler<dim, dim>               &mg_dof_pressure,
      std::vector<internal::CSRMatrix<Number>> &transfer_matrix);

    /**
     * Internal function to fill copy_indice.
     */
    void
    fill_copy_indices(const DoFHandler<dim> &mg_dof);

    /**
     * Internal function to transfer data.
     */
    template <typename VectorType, typename VectorType2>
    void
    copy_to_device(VectorType &device, const VectorType2 &host);

    /**
     * Perform the loop_body(prolongation/restriction) operation.
     */
    template <template <int, int, typename> class loop_body, int degree>
    void
    coarse_cell_loop(
      const unsigned int                                             fine_level,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const;

    void
    set_mg_constrained_dofs(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec,
      unsigned int                                                   level,
      Number                                                         val) const;
  };

  template <typename Number, typename Number2>
  __global__ void
  copy_with_indices_kernel(Number             *dst,
                           Number2            *src,
                           const unsigned int *dst_indices,
                           const unsigned int *src_indices,
                           int                 n)
  {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
      {
        dst[dst_indices[i]] = src[src_indices[i]];
      }
  }

  template <typename Number, typename Number2>
  void
  copy_with_indices(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>        &dst,
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &src,
    const CudaVector<unsigned int> &dst_indices,
    const CudaVector<unsigned int> &src_indices)
  {
    const int  n         = dst_indices.size();
    const int  blocksize = 256;
    const dim3 block_dim = dim3(blocksize);
    const dim3 grid_dim  = dim3(1 + (n - 1) / blocksize);
    copy_with_indices_kernel<<<grid_dim, block_dim>>>(dst.get_values(),
                                                      src.get_values(),
                                                      dst_indices.get_values(),
                                                      src_indices.get_values(),
                                                      n);
    AssertCudaKernel();
  }

  namespace internal
  { // Sets up most of the internal data structures of the
    // MGTransferCUDA class
    template <int dim, typename Number>
    void
    setup_transfer(
      const DoFHandler<dim>   &dof_handler_velocity,
      const DoFHandler<dim>   &dof_handler_pressure,
      const MGConstrainedDoFs *mg_constrained_dofs,
      const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
                                             &external_partitioners,
      ElementInfo<Number>                    &elem_info,
      std::vector<std::vector<unsigned int>> &level_dof_indices_parent,
      std::vector<std::vector<unsigned int>> &level_dof_indices_child,
      std::vector<unsigned int>              &n_owned_level_cells,
      std::vector<std::vector<Number>>       &weights_on_refined,
      std::vector<Table<2, unsigned int>>    &copy_indices_global_mine,
      MGLevelObject<std::shared_ptr<const Utilities::MPI::Partitioner>>
        &target_partitioners)
    {
      level_dof_indices_parent.clear();
      level_dof_indices_child.clear();
      n_owned_level_cells.clear();
      weights_on_refined.clear();

      // we collect all child DoFs of a mother cell together. For faster
      // tensorized operations, we align the degrees of freedom
      // lexicographically. We distinguish FE_Q elements and FE_DGQ elements

      const ::Triangulation<dim> &tria =
        dof_handler_velocity.get_triangulation();

      // ---------- 1. Extract info about the finite element
      elem_info.reinit(dof_handler_velocity);

      // ---------- 2. Extract and match dof indices between child and parent
      const unsigned int n_levels = tria.n_global_levels();
      level_dof_indices_parent.resize(n_levels);
      level_dof_indices_child.resize(n_levels);
      n_owned_level_cells.resize(n_levels - 1);

      const unsigned int n_child_cell_dofs = elem_info.n_child_cell_dofs;

      std::vector<types::global_dof_index> local_dof_indices_v(
        dof_handler_velocity.get_fe().n_dofs_per_cell());

      std::vector<types::global_dof_index> local_dof_indices_p(
        dof_handler_pressure.get_fe().n_dofs_per_cell());

      AssertDimension(target_partitioners.max_level(), n_levels - 1);
      Assert(external_partitioners.empty() ||
               external_partitioners.size() == n_levels,
             ExcDimensionMismatch(external_partitioners.size(), n_levels));

      for (unsigned int level = n_levels - 1; level > 0; --level)
        {
          unsigned int                         counter = 0;
          std::vector<types::global_dof_index> child_level_dof_indices;
          std::vector<types::global_dof_index> ghosted_level_dofs;

          // step 2.1: loop over the cells on the coarse side
          typename ::DoFHandler<dim>::cell_iterator
            cell_v = dof_handler_velocity.begin(level - 1),
            cell_p = dof_handler_pressure.begin(level - 1),
            endc_v = dof_handler_velocity.end(level - 1);
          for (; cell_v != endc_v; ++cell_v, ++cell_p)
            {
              // need to look into a cell if it has children and it is locally
              // owned
              if (!cell_v->has_children())
                continue;

              bool consider_cell = (tria.locally_owned_subdomain() ==
                                      numbers::invalid_subdomain_id ||
                                    cell_v->level_subdomain_id() ==
                                      tria.locally_owned_subdomain());

              if (!consider_cell)
                continue;

              counter++;

              // step 2.2: loop through children and append the dof indices to
              // the appropriate list. We need separate lists for the owned
              // coarse cell case (which will be part of
              // restriction/prolongation between level-1 and level) and the
              // remote case (which needs to store DoF indices for the
              // operations between level and level+1).
              AssertDimension(cell_v->n_children(),
                              GeometryInfo<dim>::max_children_per_cell);

              // const std::size_t start_index =
              // parent_level_dof_indices.size();
              // parent_level_dof_indices.resize(
              //   start_index + dof_handler.get_fe().n_dofs_per_cell());
              // for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
              //   parent_level_dof_indices[start_index + i] =
              //     local_dof_indices[i];

              cell_v->get_mg_dof_indices(local_dof_indices_v);
              cell_p->get_mg_dof_indices(local_dof_indices_p);
              for (auto ind : local_dof_indices_v)
                level_dof_indices_parent[level - 1].push_back(ind);
              for (auto ind : local_dof_indices_p)
                level_dof_indices_parent[level - 1].push_back(
                  dof_handler_velocity.n_dofs(level - 1) + ind);

              std::unordered_set<unsigned int> s;
              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
                {
                  if (!consider_cell)
                    continue;
                  cell_v->child(c)->get_mg_dof_indices(local_dof_indices_v);

                  resolve_identity_constraints(mg_constrained_dofs,
                                               level,
                                               local_dof_indices_v);

                  const IndexSet &owned_level_dofs =
                    dof_handler_velocity.locally_owned_mg_dofs(level);
                  for (const auto local_dof_index : local_dof_indices_v)
                    if (!owned_level_dofs.is_element(local_dof_index))
                      ghosted_level_dofs.push_back(local_dof_index);

                  for (auto ind : local_dof_indices_v)
                    if (s.count(ind) == 0)
                      {
                        child_level_dof_indices.push_back(ind);
                        s.insert(ind);
                      }
                }

              for (unsigned int c = 0;
                   c < GeometryInfo<dim>::max_children_per_cell;
                   ++c)
                {
                  if (!consider_cell)
                    continue;
                  cell_p->child(c)->get_mg_dof_indices(local_dof_indices_p);

                  // const IndexSet &owned_level_dofs =
                  //   dof_handler_pressure.locally_owned_mg_dofs(level);
                  // for (const auto local_dof_index : local_dof_indices_p)
                  //   if (!owned_level_dofs.is_element(local_dof_index))
                  //     ghosted_level_dofs.push_back(local_dof_index);

                  for (auto ind : local_dof_indices_p)
                    child_level_dof_indices.push_back(
                      dof_handler_velocity.n_dofs(level) + ind);
                }
              // n_child_cell_dofs = s.size();
            }
          n_owned_level_cells[level - 1] = counter;

          reinit_level_partitioner(dof_handler_velocity.locally_owned_mg_dofs(
                                     level),
                                   ghosted_level_dofs,
                                   external_partitioners.empty() ?
                                     nullptr :
                                     external_partitioners[level],
                                   tria.get_communicator(),
                                   target_partitioners[level],
                                   copy_indices_global_mine[level]);

          copy_indices_to_mpi_local_numbers(*target_partitioners[level],
                                            child_level_dof_indices,
                                            level_dof_indices_child[level]);
        }

      // for (auto &level_dof_indices : level_dof_indices_child)
      //   {
      //     for (auto ind : level_dof_indices)
      //       std::cout << ind << " ";
      //     std::cout << std::endl;
      //   }

      // ----------- 3. compute weights to make restriction additive

      // get the valence of the individual components and compute the weights
      // as the inverse of the valence
      weights_on_refined.resize(n_levels - 1);
      for (unsigned int level = 1; level < n_levels; ++level)
        {
          LinearAlgebra::distributed::Vector<Number> touch_count(
            dof_handler_velocity.n_dofs(level) +
            dof_handler_pressure.n_dofs(level));
          for (unsigned int c = 0; c < n_owned_level_cells[level - 1]; ++c)
            for (unsigned int j = 0; j < elem_info.n_child_cell_dofs; ++j)
              touch_count[level_dof_indices_child
                            [level][elem_info.n_child_cell_dofs * c + j]] +=
                Number(1.);
          touch_count.compress(VectorOperation::add);
          touch_count.update_ghost_values();

          weights_on_refined[level - 1].resize(n_owned_level_cells[level - 1] *
                                               elem_info.n_child_cell_dofs);
          for (unsigned int c = 0; c < n_owned_level_cells[level - 1]; ++c)
            for (unsigned int j = 0; j < n_child_cell_dofs; ++j)
              {
                weights_on_refined[level - 1][c * n_child_cell_dofs + j] =
                  Number(1.) /
                  touch_count[level_dof_indices_child
                                [level][elem_info.n_child_cell_dofs * c + j]];
              }
        }
    }
  } // namespace internal

} // namespace PSMF


// #include "cuda_mg_transfer.template.cuh"

/**
 * \page cuda_mg_transfer
 * \include cuda_mg_transfer.cuh
 */

#endif // MG_TRANSFER_CUH