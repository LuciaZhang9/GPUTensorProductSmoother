#ifndef PATCHSMOOTHER_TRANSFER_INTERNAL_H
#define PATCHSMOOTHER_TRANSFER_INTERNAL_H

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <unordered_set>

using namespace dealii;

namespace PSMF::internal
{

  template <typename Number>
  struct COOEntry
  {
    unsigned int row;
    unsigned int col;
    Number       value;
  };

  template <typename Number>
  struct CSRMatrix
  {
    std::vector<unsigned int> row_ptr;
    std::vector<unsigned int> col_idx;
    std::vector<Number>       values;
  };

  /**
   * A structure that stores data related to the finite element contained in
   * the DoFHandler. Used only for the initialization using setup_transfer.
   */
  template <typename Number>
  struct ElementInfo
  {
    unsigned int fe_degree             = 0;
    bool         element_is_continuous = 0;
    unsigned int n_components          = 0;
    unsigned int n_child_cell_dofs     = 0;

    std::vector<unsigned int> lexicographic_numbering{};
    std::vector<Number>       prolongation_matrix{};

    template <int dim>
    void
    reinit(const DoFHandler<dim> &dof_handler)
    {
      fe_degree             = dof_handler.get_fe().degree - 1;
      element_is_continuous = dof_handler.get_fe().n_dofs_per_vertex() > 0;
      n_components          = dof_handler.get_fe().n_components();

      const unsigned int n_coarse =
        dim * std::pow(fe_degree + 1, dim - 1) * (fe_degree + 2) +
        std::pow(fe_degree + 1, dim);
      n_child_cell_dofs =
        dim * std::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3) +
        std::pow(2 * fe_degree + 2, dim);
    }
  };

  // initialize the vectors needed for the transfer (and merge with the
  // content in copy_indices_global_mine)
  void
  reinit_level_partitioner(
    const IndexSet                       &locally_owned,
    std::vector<types::global_dof_index> &ghosted_level_dofs,
    const std::shared_ptr<const Utilities::MPI::Partitioner>
                                                       &external_partitioner,
    const MPI_Comm                                     &communicator,
    std::shared_ptr<const Utilities::MPI::Partitioner> &target_partitioner,
    Table<2, unsigned int> &copy_indices_global_mine);

  // Transform the ghost indices to local index space for the vector
  void
  copy_indices_to_mpi_local_numbers(
    const Utilities::MPI::Partitioner          &part,
    const std::vector<types::global_dof_index> &mine,
    std::vector<unsigned int>                  &localized_indices);

  void
  resolve_identity_constraints(
    const MGConstrainedDoFs              *mg_constrained_dofs,
    const unsigned int                    level,
    std::vector<types::global_dof_index> &dof_indices);

} // namespace PSMF::internal


#endif // PATCHSMOOTHER_TRANSFER_INTERNAL_H