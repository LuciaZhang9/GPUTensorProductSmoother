/**
 * @file laplace_operator.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the Laplace operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include <deal.II/fe/fe_interface_values.h>

#include "TPSS/move_to_deal_ii.h"
#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  struct LocalLaplace
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative, local_bilaplace
      shared_mem +=
        3 * patch_per_block * n_dofs_1d * n_dofs_1d * dim * sizeof(Number);
      // temp
      shared_mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_basic<dim, fe_degree, Number>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_basic<dim, fe_degree, Number>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::MatrixStruct>
  {
    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      static constexpr unsigned int n_patch_dofs_rt =
        dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);

      static constexpr unsigned int n_patch_dofs_dg =
        Util::pow(2 * fe_degree + 2, dim);

      static constexpr unsigned int n_patch_dofs =
        n_patch_dofs_rt + n_patch_dofs_dg;

      // local_src, local_dst
      shared_mem += 2 * patch_per_block * n_patch_dofs * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_matrix<dim, fe_degree, Number>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));

      block_dim = 256;
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_matrix<dim, fe_degree, Number>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::Basic>
  {
    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      static constexpr unsigned int n_patch_dofs_rt =
        dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);

      static constexpr unsigned int n_patch_dofs_dg =
        Util::pow(2 * fe_degree + 2, dim);

      static constexpr unsigned int n_patch_dofs =
        n_patch_dofs_rt + n_patch_dofs_dg;

      // local_src, local_dst
      shared_mem += 2 * patch_per_block * n_patch_dofs * sizeof(Number);

      // tmp
      shared_mem += (dim - 1) * patch_per_block * n_patch_dofs * sizeof(Number);

      // L M
      shared_mem += dim * patch_per_block * dim * 2 *
                    Util::pow(2 * fe_degree + 3, 2) * sizeof(Number);
      // M D
      shared_mem += patch_per_block * Util::pow(2 * fe_degree + 3, 2) * dim *
                    dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_basic<dim, fe_degree, Number>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));

      block_dim =
        dim3(2 * fe_degree + 3, patch_per_block * dim * (2 * fe_degree + 3));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_basic<dim, fe_degree, Number>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };


  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  class LaplaceOperator : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceOperator()
    {}

    void
    initialize(
      std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data_,
      const DoFHandler<dim>                                          &mg_dof,
      const unsigned int                                              level)
    {
      data          = data_;
      dof_handler   = &mg_dof;
      dof_handler_v = &mg_dof;
      mg_level      = level;
    }

    void
    initialize(
      std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data_,
      const DoFHandler<dim>                                          &mg_dof,
      const DoFHandler<dim>                                          &mg_dof_v,
      const unsigned int                                              level)
    {
      data          = data_;
      dof_handler   = &mg_dof;
      dof_handler_v = &mg_dof_v;
      mg_level      = level;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      auto tmp = src;

      data->set_constrained_values(tmp);

      LocalLaplace<dim, fe_degree, Number, kernel> local_laplace;

      data->cell_loop(local_laplace, tmp, dst);

      data->copy_constrained_values(src, dst); //
    }

    void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
             &src) const
    {
      vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const
    {
      const unsigned int n_dofs = dof_handler->n_dofs(mg_level);
      vec.reinit(n_dofs);
    }

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
      const Function<dim, Number> &exact_solution,
      const unsigned int           mg_level) const
    {
      dst = 0.;
      // src.update_ghost_values();

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);

      AffineConstraints<Number> constraints;
      constraints.clear();
      VectorToolsFix::project_boundary_values_div_conforming(
        *dof_handler_v, 0, exact_solution, 0, constraints, MappingQ1<dim>());
      constraints.close();

      const QGauss<dim>      quadrature_formula(fe_degree + 2);
      FEValues<dim>          fe_values(dof_handler_v->get_fe(),
                              quadrature_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values);
      FEInterfaceValues<dim> fe_interface_values(
        dof_handler_v->get_fe(),
        QGauss<dim - 1>(fe_degree + 2),
        update_values | update_gradients | update_quadrature_points |
          update_hessians | update_JxW_values | update_normal_vectors);

      const unsigned int dofs_per_cell =
        dof_handler_v->get_fe().n_dofs_per_cell();

      const unsigned int        n_q_points = quadrature_formula.size();
      Vector<Number>            cell_rhs(dofs_per_cell);
      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      auto begin = dof_handler_v->begin_mg(mg_level);
      auto end   = dof_handler_v->end_mg(mg_level);

      const FEValuesExtractors::Vector velocities(0);

      for (auto cell = begin; cell != end; ++cell)
        if (cell->is_locally_owned_on_level())
          {
            cell_rhs = 0;
            fe_values.reinit(cell);

            std::vector<Tensor<1, dim>> load_values;
            const auto &q_points = fe_values.get_quadrature_points();
            std::transform(q_points.cbegin(),
                           q_points.cend(),
                           std::back_inserter(load_values),
                           [&](const auto &x_q) {
                             Tensor<1, dim> value;
                             for (auto c = 0U; c < dim; ++c)
                               value[c] = rhs_function.value(x_q, c);
                             return value;
                           });

            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  cell_rhs(i) +=
                    (fe_values[velocities].value(i, q_index) *
                     load_values[q_index] * fe_values.JxW(q_index));
              }

            // std::cout << fe_values[velocities].value(0, 6) << " " <<
            // load_values[5] << " " << fe_values.JxW(3) << std::endl;

            cell->get_mg_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs_host);
          }

      for (auto cell = begin; cell != end; ++cell)
        if (cell->is_locally_owned_on_level())
          {
            for (const unsigned int face_no : cell->face_indices())
              if (cell->at_boundary(face_no))
                {
                  fe_interface_values.reinit(cell, face_no);

                  const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                  Vector<Number> cell_rhs_face(n_interface_dofs);
                  cell_rhs_face = 0;

                  const auto &q_points =
                    fe_interface_values.get_quadrature_points();
                  const std::vector<double> &JxW =
                    fe_interface_values.get_JxW_values();
                  const std::vector<Tensor<1, dim>> &normals =
                    fe_interface_values.get_normal_vectors();

                  std::vector<Tensor<1, dim>> tangential_solution_values;
                  std::vector<Tensor<1, dim>> solution_values;
                  std::transform(q_points.cbegin(),
                                 q_points.cend(),
                                 std::back_inserter(solution_values),
                                 [&](const auto &x_q) {
                                   Tensor<1, dim> value;
                                   for (auto c = 0U; c < dim; ++c)
                                     value[c] = exact_solution.value(x_q, c);
                                   return value;
                                 });
                  std::transform(solution_values.cbegin(),
                                 solution_values.cend(),
                                 normals.cbegin(),
                                 std::back_inserter(tangential_solution_values),
                                 [](const auto &u_q, const auto &normal) {
                                   return u_q - ((u_q * normal) * normal);
                                 });

                  const unsigned int p = fe_degree;
                  const auto         h = cell->extent_in_direction(
                    GeometryInfo<dim>::unit_normal_direction[face_no]);
                  const auto   one_over_h   = (0.5 / h) + (0.5 / h);
                  const auto   gamma        = p == 0 ? 1 : p * (p + 1);
                  const double gamma_over_h = 2.0 * gamma * one_over_h;

                  for (unsigned int qpoint = 0; qpoint < q_points.size();
                       ++qpoint)
                    {
                      const auto &n = normals[qpoint];

                      for (unsigned int i = 0; i < n_interface_dofs; ++i)
                        {
                          const auto av_gradients_i_dot_n_dot_n =
                            (fe_interface_values.average_of_shape_gradients(
                               i, qpoint) *
                             n * n);
                          const auto jump_val_i_dot_n =
                            (fe_interface_values.jump_in_shape_values(i,
                                                                      qpoint) *
                             n);
                          cell_rhs_face(i) +=
                            (-av_gradients_i_dot_n_dot_n * // - {grad v n n }
                               (tangential_solution_values
                                  [qpoint])                //   (u_exact
                                                           //   . n)
                             +                             // +
                             gamma_over_h                  //  gamma/h
                               * jump_val_i_dot_n          // [v n]
                               *
                               (tangential_solution_values[qpoint]) // (u_exact
                                                                    // . n)
                             ) *
                            JxW[qpoint];                            // dx
                        }
                    }

                  auto dof_indices =
                    fe_interface_values.get_interface_dof_indices();
                  constraints.distribute_local_to_global(cell_rhs_face,
                                                         dof_indices,
                                                         system_rhs_host);
                }
          }

      system_rhs_host.compress(VectorOperation::add);
      rw_vector.import(system_rhs_host, VectorOperation::insert);
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      // system_rhs_host = 0.;
      // system_rhs_host[10] = 1.;
      // rw_vector.import(system_rhs_host, VectorOperation::insert);
      // system_rhs_dev.import(rw_vector, VectorOperation::insert);

      // vmult(dst, system_rhs_dev);
      // system_rhs_host.print(std::cout);

      vmult(dst, src);
      dst.sadd(-1., system_rhs_dev);
    }

    unsigned int
    get_mg_level() const
    {
      return mg_level;
    }

    const DoFHandler<dim> *
    get_dof_handler() const
    {
      return dof_handler;
    }

    std::size_t
    memory_consumption() const
    {
      std::size_t result = sizeof(*this);
      return result;
    }

  private:
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;

    const DoFHandler<dim> *dof_handler;
    const DoFHandler<dim> *dof_handler_v;
    unsigned int           mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH