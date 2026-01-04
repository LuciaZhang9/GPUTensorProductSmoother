#include "transfer_internal.h"

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_transfer_internal.h>

namespace PSMF::internal
{


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
    Table<2, unsigned int> &copy_indices_global_mine)
  {
    std::sort(ghosted_level_dofs.begin(), ghosted_level_dofs.end());
    IndexSet ghosted_dofs(locally_owned.size());
    ghosted_dofs.add_indices(ghosted_level_dofs.begin(),
                             std::unique(ghosted_level_dofs.begin(),
                                         ghosted_level_dofs.end()));
    ghosted_dofs.compress();

    // Add possible ghosts from the previous content in the vector
    if (target_partitioner.get() != nullptr &&
        target_partitioner->size() == locally_owned.size())
      {
        ghosted_dofs.add_indices(target_partitioner->ghost_indices());
      }

    // check if the given partitioner's ghosts represent a superset of the
    // ghosts we require in this function
    const int ghosts_locally_contained =
      (external_partitioner.get() != nullptr &&
       (external_partitioner->ghost_indices() & ghosted_dofs) == ghosted_dofs) ?
        1 :
        0;
    if (external_partitioner.get() != nullptr &&
        Utilities::MPI::min(ghosts_locally_contained, communicator) == 1)
      {
        // shift the local number of the copy indices according to the new
        // partitioner that we are going to use during the access to the
        // entries
        if (target_partitioner.get() != nullptr &&
            target_partitioner->size() == locally_owned.size())
          for (unsigned int i = 0; i < copy_indices_global_mine.n_cols(); ++i)
            copy_indices_global_mine(1, i) =
              external_partitioner->global_to_local(
                target_partitioner->local_to_global(
                  copy_indices_global_mine(1, i)));
        target_partitioner = external_partitioner;
      }
    else
      {
        if (target_partitioner.get() != nullptr &&
            target_partitioner->size() == locally_owned.size())
          for (unsigned int i = 0; i < copy_indices_global_mine.n_cols(); ++i)
            copy_indices_global_mine(1, i) =
              locally_owned.n_elements() +
              ghosted_dofs.index_within_set(target_partitioner->local_to_global(
                copy_indices_global_mine(1, i)));
        target_partitioner.reset(new Utilities::MPI::Partitioner(locally_owned,
                                                                 ghosted_dofs,
                                                                 communicator));
      }
  }

  // Transform the ghost indices to local index space for the vector
  void
  copy_indices_to_mpi_local_numbers(
    const Utilities::MPI::Partitioner          &part,
    const std::vector<types::global_dof_index> &mine,
    std::vector<unsigned int>                  &localized_indices)
  {
    localized_indices.resize(mine.size(), numbers::invalid_unsigned_int);

    for (unsigned int i = 0; i < mine.size(); ++i)
      if (mine[i] != numbers::invalid_dof_index)
        localized_indices[i] = mine[i];

    (void)part;
    // for (unsigned int i = 0; i < mine.size(); ++i)
    //   if (mine[i] != numbers::invalid_dof_index)
    //     localized_indices[i] = part.global_to_local(mine[i]);
  }

  void
  resolve_identity_constraints(
    const MGConstrainedDoFs              *mg_constrained_dofs,
    const unsigned int                    level,
    std::vector<types::global_dof_index> &dof_indices)
  {
    if (mg_constrained_dofs != nullptr &&
        mg_constrained_dofs->get_level_constraints(level).n_constraints() > 0)
      for (auto &ind : dof_indices)
        if (mg_constrained_dofs->get_level_constraints(level)
              .is_identity_constrained(ind))
          {
            Assert(mg_constrained_dofs->get_level_constraints(level)
                       .get_constraint_entries(ind)
                       ->size() == 1,
                   ExcInternalError());
            ind = mg_constrained_dofs->get_level_constraints(level)
                    .get_constraint_entries(ind)
                    ->front()
                    .first;
          }
  }

} // namespace PSMF::internal