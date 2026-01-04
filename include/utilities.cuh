/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of helper functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>

#include <execution>
#include <vector>

using namespace dealii;

namespace Util
{
  template <typename T>
  __host__ __device__ constexpr T
  pow(const T base, const int iexp)
  {
    return iexp <= 0 ?
             1 :
             (iexp == 1 ?
                base :
                (((iexp % 2 == 1) ? base : 1) * pow(base * base, iexp / 2)));
  }


  constexpr unsigned int padding   = 0;
  constexpr unsigned int BLOCK_DIM = 256;

  constexpr unsigned int DIM        = 3;
  constexpr unsigned int MAX_DEGREE = 3;
  constexpr unsigned int MAX_PATCH_DOFS_RT =
    DIM * pow(2 * MAX_DEGREE + 2, DIM - 1) * (2 * MAX_DEGREE + 3);
  constexpr unsigned int MAX_PATCH_DOFS_RT_INT =
    DIM * pow(2 * MAX_DEGREE + 2, DIM - 1) * (2 * MAX_DEGREE + 1);
  constexpr unsigned int MAX_PATCH_DOFS_DG = pow(2 * MAX_DEGREE + 2, DIM);

  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 12 13 | 14 15
   *  8  9 | 10 11
   * ------|------
   *  4  5 |  6  7
   *  0  1 |  2  3
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices(unsigned int *first_dofs,
                  unsigned int  local_patch,
                  unsigned int  local_tid_x,
                  unsigned int  tid_y,
                  unsigned int  tid_z)
  {
    const unsigned int z_off = tid_z / (fe_degree + 1);
    const unsigned int y_off = tid_y / (fe_degree + 1);
    const unsigned int x_off = local_tid_x / (fe_degree + 1);
    const unsigned int z     = tid_z % (fe_degree + 1);
    const unsigned int y     = tid_y % (fe_degree + 1);
    const unsigned int x     = local_tid_x % (fe_degree + 1);

    return first_dofs[z_off * 4 + y_off * 2 + x_off] +
           z * (fe_degree + 1) * (fe_degree + 1) + y * (fe_degree + 1) + x;
  }


  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 10 11 | 14 15
   *  8  9 | 12 13
   * ------|------
   *  2  3 |  6  7
   *  0  1 |  4  5
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices_cell(unsigned int *first_dofs, unsigned int linear_tid)
  {
    constexpr unsigned int cell_dofs = pow(fe_degree + 1, dim);

    const unsigned int cell           = linear_tid / cell_dofs;
    const unsigned int local_cell_tid = linear_tid % cell_dofs;

    return first_dofs[cell] + local_cell_tid;
  }

  /**
   * Point-wise comparator for renumbering global dofs
   * from hierarchy to lexicographic ordering.
   * @tparam dim dimension
   */
  template <int dim>
  struct ComparePointwiseLexicographic;

  template <>
  struct ComparePointwiseLexicographic<1>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<1>, types::global_dof_index> &c1,
               const std::pair<Point<1>, types::global_dof_index> &c2) const
    {
      return c1.first[0] < c2.first[0];
    }
  };

  template <>
  struct ComparePointwiseLexicographic<2>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<2>, types::global_dof_index> &c1,
               const std::pair<Point<2>, types::global_dof_index> &c2) const
    {
      const double y_err = std::abs(c1.first[1] - c2.first[1]);

      if (y_err > 1e-10 && c1.first[1] < c2.first[1])
        return true;
      // y0 == y1
      if (y_err < 1e-10 && c1.first[0] < c2.first[0])
        return true;
      return false;
    }
  };

  template <>
  struct ComparePointwiseLexicographic<3>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<3>, types::global_dof_index> &c1,
               const std::pair<Point<3>, types::global_dof_index> &c2) const
    {
      const double z_err = std::abs(c1.first[2] - c2.first[2]);
      const double y_err = std::abs(c1.first[1] - c2.first[1]);

      if (z_err > 1e-10 && c1.first[2] < c2.first[2])
        return true;
      // z0 == z1
      if (z_err < 1e-10 && y_err > 1e-10 && c1.first[1] < c2.first[1])
        return true;
      // z0 == z1, y0 == y1
      if (z_err < 1e-10 && y_err < 1e-10 && c1.first[0] < c2.first[0])
        return true;

      return false;
    }
  };

  /**
   * Compute the set of renumbering indices on finest level needed by the
   * Lexicographic() function. Does not perform the renumbering on the
   * DoFHandler dofs but returns the renumbering vector.
   *
   * Using @b DoFHandler<dim, spacedim>::active_cell_iterators loop all cells.
   * @tparam dim
   * @tparam spacedim
   * @param new_indices
   * @param dof
   */
  template <int dim, int spacedim>
  void
  compute_Lexicographic(std::vector<types::global_dof_index> &new_indices,
                        const DoFHandler<dim, spacedim>      &dof)
  {
    Assert((dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
              &dof.get_triangulation()) == nullptr),
           ExcNotImplemented());

    const unsigned int                                    n_dofs = dof.n_dofs();
    std::vector<std::pair<Point<spacedim>, unsigned int>> support_point_list(
      n_dofs);

    const hp::FECollection<dim> &fe_collection = dof.get_fe_collection();
    Assert(fe_collection[0].has_support_points(),
           typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
    hp::QCollection<dim> quadrature_collection;
    for (unsigned int comp = 0; comp < fe_collection.size(); ++comp)
      {
        Assert(fe_collection[comp].has_support_points(),
               typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
        quadrature_collection.push_back(
          Quadrature<dim>(fe_collection[comp].get_unit_support_points()));
      }
    hp::FEValues<dim, spacedim> hp_fe_values(fe_collection,
                                             quadrature_collection,
                                             update_quadrature_points);

    std::vector<bool> already_touched(n_dofs, false);

    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
        local_dof_indices.resize(dofs_per_cell);
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        cell->get_active_or_mg_dof_indices(local_dof_indices);
        const std::vector<Point<spacedim>> &points =
          fe_values.get_quadrature_points();
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (!already_touched[local_dof_indices[i]])
            {
              support_point_list[local_dof_indices[i]].first = points[i];
              support_point_list[local_dof_indices[i]].second =
                local_dof_indices[i];
              already_touched[local_dof_indices[i]] = true;
            }
      }

    ComparePointwiseLexicographic<spacedim> comparator;
    std::sort(std::execution::par,
              support_point_list.begin(),
              support_point_list.end(),
              comparator);
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
      new_indices[support_point_list[i].second] = i;
  }
  /**
   * Compute the set of renumbering indices on one level of a multigrid
   * hierarchy needed by the Lexicographic() function. Does not perform the
   * renumbering on the DoFHandler dofs but returns the renumbering vector.
   *
   * Using @b DoFHandler<dim, spacedim>::level_cell_iterator loop all cells.
   * @tparam dim
   * @tparam spacedim
   * @param new_indices
   * @param dof
   * @param level
   */
  template <int dim, int spacedim>
  void
  compute_Lexicographic(std::vector<types::global_dof_index> &new_indices,
                        const DoFHandler<dim, spacedim>      &dof,
                        const unsigned int                    level)
  {
    Assert(dof.get_fe().has_support_points(),
           typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
    const unsigned int n_dofs = dof.n_dofs(level);
    std::vector<std::pair<Point<spacedim>, unsigned int>> support_point_list(
      n_dofs);

    Quadrature<dim>         q_dummy(dof.get_fe().get_unit_support_points());
    FEValues<dim, spacedim> fe_values(dof.get_fe(),
                                      q_dummy,
                                      update_quadrature_points);

    std::vector<bool> already_touched(dof.n_dofs(), false);

    const unsigned int dofs_per_cell = dof.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    typename DoFHandler<dim, spacedim>::level_cell_iterator begin =
      dof.begin(level);
    typename DoFHandler<dim, spacedim>::level_cell_iterator end =
      dof.end(level);
    for (; begin != end; ++begin)
      {
        const typename Triangulation<dim, spacedim>::cell_iterator &begin_tria =
          begin;
        begin->get_active_or_mg_dof_indices(local_dof_indices);
        fe_values.reinit(begin_tria);
        const std::vector<Point<spacedim>> &points =
          fe_values.get_quadrature_points();
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (!already_touched[local_dof_indices[i]])
            {
              support_point_list[local_dof_indices[i]].first = points[i];
              support_point_list[local_dof_indices[i]].second =
                local_dof_indices[i];
              already_touched[local_dof_indices[i]] = true;
            }
      }
    ComparePointwiseLexicographic<spacedim> comparator;
    std::sort(std::execution::par,
              support_point_list.begin(),
              support_point_list.end(),
              comparator);
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
      new_indices[support_point_list[i].second] = i;
  }

  /**
   * Lexicographic numbering on finest level.
   * @tparam dim
   * @tparam spacedim
   * @param dof
   */
  template <int dim, int spacedim>
  void
  Lexicographic(DoFHandler<dim, spacedim> &dof)
  {
    std::vector<types::global_dof_index> renumbering(dof.n_dofs());
    compute_Lexicographic(renumbering, dof);
    dof.renumber_dofs(renumbering);
  }
  /**
   * Lexicographic numbering on one level of a multigrid hierarchy.
   * @tparam dim
   * @tparam spacedim
   * @param dof
   * @param level
   */
  template <int dim, int spacedim>
  void
  Lexicographic(DoFHandler<dim, spacedim> &dof, const unsigned int level)
  {
    std::vector<types::global_dof_index> renumbering(dof.n_dofs(level));
    compute_Lexicographic(renumbering, dof, level);
    dof.renumber_dofs(level, renumbering);
  }

} // namespace Util

#endif // UTILITIES_CUH