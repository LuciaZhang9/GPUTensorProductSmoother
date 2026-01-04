/**
 * @file benchmark_mg.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Benchmarks on different MG componts.
 * @version 1.0
 * @date 2023-01-02
 *
 * @copyright Copyright (c) 2023
 *
 */


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "cuda_mg_transfer.cuh"
#include "equation_data.h"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

using namespace dealii;

#define MODE 0
// 0 - ncu
// 1 - perf

template <int dim>
using Solution = Stokes::NoSlipExp::Solution<dim>;

template <int dim>
using SolutionVelocity = Stokes::NoSlipExp::SolutionVelocity<dim>;

template <int dim>
using SolutionPressure = Stokes::NoSlipExp::SolutionPressure<dim>;

template <int dim>
using RightHandSide = Stokes::ManufacturedLoad<dim>;

template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  using full_number   = double;
  using vcycle_number = float;
  using MatrixFreeDP  = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
  using MatrixFreeSP  = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;

  using VectorTypeDP =
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>;
  using VectorTypeSP =
    LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::CUDA>;

  using VectorTypeDPHost = Vector<full_number>;

  LaplaceProblem();
  ~LaplaceProblem();
  void
  run();

private:
  void
  setup_system();
  void
  assemble_rhs();
  void
  bench_Ax();
  void
  bench_transfer();
  void
  bench_smooth();

  template <PSMF::LaplaceVariant kernel>
  void
  do_Ax();
  template <PSMF::LocalSolverVariant local_solver,
            PSMF::LaplaceVariant     smooth_vmult,
            PSMF::SmootherVariant    smooth_inv>
  void
  do_smooth();
  size_t
  disp_gpu_usage();

  Triangulation<dim>                  triangulation;
  std::shared_ptr<FiniteElement<dim>> fe;
  DoFHandler<dim>                     dof_handler;
  DoFHandler<dim>                     dof_handler_velocity;
  DoFHandler<dim>                     dof_handler_pressure;
  MappingQ1<dim>                      mapping;

  MGConstrainedDoFs mg_constrained_dofs;

  std::shared_ptr<MatrixFreeDP> mfdata_dp;
  std::shared_ptr<MatrixFreeSP> mfdata_sp;

  VectorTypeDP solution_dp;
  VectorTypeDP system_rhs_dp;

  VectorTypeDP system_rhs_dev;

  VectorTypeSP solution_sp;
  VectorTypeSP system_rhs_sp;

  double base_time_dp;
  double base_time_sp;

  unsigned int N;
  unsigned int n_mv;
  unsigned int n_dofs;
  unsigned int maxlevel;

  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;

  std::array<ConvergenceTable, 5> info_table;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
  , fe([&]() -> std::shared_ptr<FiniteElement<dim>> {
    if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::Q)
      return std::make_shared<FE_Q<dim>>(fe_degree);
    else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::RT)
      return std::make_shared<FESystem<dim>>(FE_RaviartThomas_new<dim>(
                                               fe_degree),
                                             1,
                                             FE_DGQLegendre<dim>(fe_degree),
                                             1);
    return std::shared_ptr<FiniteElement<dim>>();
  }())
  , dof_handler(triangulation)
  , dof_handler_velocity(triangulation)
  , dof_handler_pressure(triangulation)
  , base_time_dp(0.)
  , base_time_sp(0.)
  , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
{
  const auto filename = Util::get_filename();
  fout.open("Benchmark_" + filename + ".log", std::ios_base::out);
  pcout = std::make_shared<ConditionalOStream>(fout, true);
}
template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::~LaplaceProblem()
{
  fout.close();
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_system()
{
  Timer time;

  dof_handler_velocity.distribute_dofs(fe->get_sub_fe(0, dim));
  dof_handler_velocity.distribute_mg_dofs();

  dof_handler_pressure.distribute_dofs(fe->get_sub_fe(dim, 1));
  dof_handler_pressure.distribute_mg_dofs();

  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();

  n_dofs = dof_handler.n_dofs();
#if MODE == 0
  N    = 1;
  n_mv = 1;
#elif MODE == 1
  N    = 5;
  n_mv = n_dofs < 10000000 ? 100 : 20;
#endif

  *pcout << "Setting up dofs...\n";

  *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
         << dof_handler_velocity.n_dofs() << " + "
         << dof_handler_pressure.n_dofs() << ")" << std::endl;

  *pcout << "DoF setup time:         " << time.wall_time() << "s" << std::endl;

  time.restart();

  *pcout << "Setting up Matrix-Free...\n";
  // Initialization of Dirichlet boundaries
  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handler_velocity);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_velocity,
                                                     dirichlet_boundary);
  MappingQ1<dim> mapping;
  maxlevel = triangulation.n_global_levels() - 1;

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                maxlevel,
                                                relevant_dofs);
  // DP
  {
    typename MatrixFreeDP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_dp = std::make_shared<MatrixFreeDP>();
    mfdata_dp->reinit(dof_handler_velocity,
                      dof_handler_pressure,
                      mg_constrained_dofs,
                      maxlevel,
                      additional_data);
  }
  // SP
  {
    typename MatrixFreeSP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_sp = std::make_shared<MatrixFreeSP>();
    mfdata_sp->reinit(dof_handler_velocity,
                      dof_handler_pressure,
                      mg_constrained_dofs,
                      maxlevel,
                      additional_data);
  }

  *pcout << "Matrix-free setup time: " << time.wall_time() << "s" << std::endl;
  *pcout << "Matrix-free Mem: " << disp_gpu_usage() << " MB" << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  Timer time;

  SolutionVelocity<dim> exact_solution;
  RightHandSide<dim>    rhs_function(std::make_shared<Solution<dim>>());

  const unsigned int n_dofs = dof_handler.n_dofs();

  system_rhs_dev.reinit(n_dofs);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    n_dofs);

  LinearAlgebra::ReadWriteVector<double> rw_vector(n_dofs);

  AffineConstraints<double> constraints;
  constraints.clear();
  VectorToolsFix::project_boundary_values_div_conforming(
    dof_handler_velocity, 0, exact_solution, 0, constraints, MappingQ1<dim>());
  constraints.close();

  const QGauss<dim>      quadrature_formula(fe_degree + 2);
  FEValues<dim>          fe_values(dof_handler_velocity.get_fe(),
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);
  FEInterfaceValues<dim> fe_interface_values(dof_handler_velocity.get_fe(),
                                             QGauss<dim - 1>(fe_degree + 2),
                                             update_values | update_gradients |
                                               update_quadrature_points |
                                               update_hessians |
                                               update_JxW_values |
                                               update_normal_vectors);

  const unsigned int dofs_per_cell =
    dof_handler_velocity.get_fe().n_dofs_per_cell();

  const unsigned int        n_q_points = quadrature_formula.size();
  Vector<double>            cell_rhs(dofs_per_cell);
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);

  auto begin = dof_handler_velocity.begin_mg(
    dof_handler.get_triangulation().n_global_levels() - 1);
  auto end = dof_handler_velocity.end_mg(
    dof_handler.get_triangulation().n_global_levels() - 1);

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
              cell_rhs(i) += (fe_values[velocities].value(i, q_index) *
                              load_values[q_index] * fe_values.JxW(q_index));
          }

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
              Vector<double> cell_rhs_face(n_interface_dofs);
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

              for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
                {
                  const auto &n = normals[qpoint];

                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    {
                      const auto av_gradients_i_dot_n_dot_n =
                        (fe_interface_values.average_of_shape_gradients(
                           i, qpoint) *
                         n * n);
                      const auto jump_val_i_dot_n =
                        (fe_interface_values.jump_in_shape_values(i, qpoint) *
                         n);
                      cell_rhs_face(i) +=
                        (-av_gradients_i_dot_n_dot_n * // - {grad v n n }
                           (tangential_solution_values[qpoint])   //   (u_exact
                                                                  //   . n)
                         +                                        // +
                         gamma_over_h                             //  gamma/h
                           * jump_val_i_dot_n                     // [v n]
                           * (tangential_solution_values[qpoint]) // (u_exact
                                                                  // . n)
                         ) *
                        JxW[qpoint]; // dx
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

  *pcout << "RHS Mem: " << disp_gpu_usage() << " MB" << std::endl;
}
template <int dim, int fe_degree>
template <PSMF::LaplaceVariant kernel>
void
LaplaceProblem<dim, fe_degree>::do_Ax()
{
  PSMF::LaplaceOperator<dim, fe_degree, full_number, kernel> matrix_dp;
  matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);
  matrix_dp.initialize_dof_vector(system_rhs_dp);
  solution_dp.reinit(system_rhs_dp);

  system_rhs_dp = 1.;
  solution_dp   = 0.;

  // std::cout << "TESTING Ax!!!\n";
  std::srand(0);
  LinearAlgebra::ReadWriteVector<full_number> rw_vector(dof_handler.n_dofs());
  for (unsigned int i = 0; i < rw_vector.size(); ++i)
    rw_vector[i] = ((double)std::rand()) / RAND_MAX;
  system_rhs_dp.import(rw_vector, VectorOperation::insert);

  // matrix_dp.vmult(solution_dp, system_rhs_dp);
  // solution_dp.print(std::cout, 4, false);

  // rw_vector     = 0;
  // system_rhs_dp = 0.;
  // for (unsigned int i = 0; i < rw_vector.size(); ++i)
  //   {
  //     rw_vector[i] = 1.;
  //     system_rhs_dp.import(rw_vector, VectorOperation::insert);
  //     matrix_dp.vmult(solution_dp, system_rhs_dp);
  //     std::cout << i << std::endl;
  //     solution_dp.print(std::cout);
  //     // std::cout << i << " " << solution_dp.l2_norm() << std::endl;
  //     rw_vector[i] = 0;
  //   }

  // std::cout << "TESTING Ax!!!\n";

  Timer  time;
  double best_time = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          matrix_dp.vmult(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  // solution_dp.print(std::cout);
  std::cout << std::setprecision(17) << solution_dp.l2_norm() << std::endl;

  info_table[0].add_value("Name", std::string(LaplaceToString(kernel)) + " DP");
  info_table[0].add_value("Time[s]", best_time);
  info_table[0].add_value("Perf[Dof/s]", n_dofs / best_time);


  PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, kernel> matrix_sp;
  matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);
  matrix_sp.initialize_dof_vector(system_rhs_sp);
  solution_sp.reinit(system_rhs_sp);

  system_rhs_sp = 1.;
  solution_sp   = 0.;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          matrix_sp.vmult(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  std::cout << std::setprecision(12) << solution_sp.l2_norm() << std::endl;

  info_table[1].add_value("Name", std::string(LaplaceToString(kernel)) + " SP");
  info_table[1].add_value("Time[s]", best_time);
  info_table[1].add_value("Perf[Dof/s]", n_dofs / best_time);

  *pcout << "Ax Mem: " << disp_gpu_usage() << " MB" << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_Ax()
{
  *pcout << "Benchmarking Mat-vec...\n";

  do_Ax<CT::LAPLACE_TYPE_[0]>();

  // for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
  //   switch (CT::LAPLACE_TYPE_[k])
  //     {
  //       case PSMF::LaplaceVariant::Basic:
  //         do_Ax<PSMF::LaplaceVariant::Basic>();
  //         break;
  //       case PSMF::LaplaceVariant::BasicCell:
  //         do_Ax<PSMF::LaplaceVariant::BasicCell>();
  //         break;
  //       case PSMF::LaplaceVariant::MatrixStruct:
  //         do_Ax<PSMF::LaplaceVariant::MatrixStruct>();
  //         break;
  //       case PSMF::LaplaceVariant::ConflictFree:
  //         do_Ax<PSMF::LaplaceVariant::ConflictFree>();
  //         break;
  //       default:
  //         AssertThrow(false, ExcMessage("Invalid Laplace Variant."));
  //     }
}

template <int dim, typename VectorType, int spacedim>
void
reinit_vector(const DoFHandler<dim, spacedim> &mg_dof, VectorType &v)
{
  for (unsigned int level = v.min_level(); level <= v.max_level(); ++level)
    {
      unsigned int n = mg_dof.n_dofs(level);
      v[level].reinit(n);
    }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_transfer()
{
  *pcout << "Benchmarking Transfer in double precision...\n";

  unsigned int max_level = triangulation.n_levels() - 1;
  VectorTypeDP u_coarse(dof_handler.n_dofs(max_level - 1));
  VectorTypeSP u_coarse_(dof_handler.n_dofs(max_level - 1));
  u_coarse  = 1.;
  u_coarse_ = 1.;

  PSMF::MGTransferCUDA<dim, full_number> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler_velocity, dof_handler_pressure);

  auto assign_vector_cuda = [](auto &vec) {
    LinearAlgebra::ReadWriteVector<double> rw_vector(vec.size());
    for (unsigned int i = 0; i < rw_vector.size(); ++i)
      rw_vector(i) = i;

    vec.import(rw_vector, VectorOperation::insert);
  };

  auto assign_vector_host = [](auto &vec, auto shift) {
    for (unsigned int i = 0; i < vec.size(); ++i)
      vec(i) = shift + i;
  };

  // check
  {
    MGLevelObject<VectorTypeDP> u(0, triangulation.n_levels() - 1);

    reinit_vector(dof_handler, u);

    const unsigned int max_level = u.max_level();

    assign_vector_cuda(u[max_level - 1]);

    // std::cout << " CUDA\n";
    mg_transfer.prolongate(max_level, u[max_level], u[max_level - 1]);
    // u[max_level].print(std::cout);

    mg_transfer.restrict_and_add(max_level, u[max_level - 1], u[max_level]);
    // u[max_level - 1].print(std::cout);

    mg_transfer.copy_from_mg(dof_handler, system_rhs_dp, u);
    // system_rhs_dp.print(std::cout);

    u[max_level] = 0;
    mg_transfer.copy_to_mg(dof_handler, u, system_rhs_dp);
    // u[max_level].print(std::cout);

    // ref
    MGTransferPrebuilt<VectorTypeDPHost> tran_v(mg_constrained_dofs);
    tran_v.build(dof_handler_velocity);

    MGTransferPrebuilt<VectorTypeDPHost> tran_p;
    tran_p.build(dof_handler_pressure);

    // std::cout << "\n HOST\n";

    // tran_v.print_matrices(std::cout);
    // tran_p.print_matrices(std::cout);

    MGLevelObject<VectorTypeDPHost> vec_v(0, triangulation.n_levels() - 1);
    reinit_vector(dof_handler_velocity, vec_v);

    MGLevelObject<VectorTypeDPHost> vec_p(0, triangulation.n_levels() - 1);
    reinit_vector(dof_handler_pressure, vec_p);

    assign_vector_host(vec_v[max_level - 1], 0);
    assign_vector_host(vec_p[max_level - 1], vec_v[max_level - 1].size());

    tran_v.prolongate(max_level, vec_v[max_level], vec_v[max_level - 1]);
    // vec_v[max_level].print(std::cout);

    tran_p.prolongate(max_level, vec_p[max_level], vec_p[max_level - 1]);
    // vec_p[max_level].print(std::cout);

    tran_v.restrict_and_add(max_level, vec_v[max_level - 1], vec_v[max_level]);
    // vec_v[max_level - 1].print(std::cout);

    tran_p.restrict_and_add(max_level, vec_p[max_level - 1], vec_p[max_level]);
    // vec_p[max_level - 1].print(std::cout);
  }

  Timer  time;
  double best_time  = 1e10;
  double best_time2 = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer.prolongate(max_level, system_rhs_dp, u_coarse);
          mg_transfer.restrict_and_add(max_level, u_coarse, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[2].add_value("Name", "Transfer DP");
  info_table[2].add_value("Time[s]", best_time);
  info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time);

  *pcout << "Benchmarking Transfer in single precision...\n";

  PSMF::MGTransferCUDA<dim, vcycle_number> mg_transfer_(mg_constrained_dofs);
  mg_transfer_.build(dof_handler_velocity, dof_handler_pressure);

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer_.prolongate(max_level, system_rhs_sp, u_coarse_);
          mg_transfer_.restrict_and_add(max_level, u_coarse_, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time2 = std::min(time.wall_time() / n_mv, best_time2);
    }

  info_table[2].add_value("Name", "Transfer SP");
  info_table[2].add_value("Time[s]", best_time2);
  info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time2);

  *pcout << "Transfer Mem: " << disp_gpu_usage() << " MB" << std::endl;
}

template <int dim, int fe_degree>
template <PSMF::LocalSolverVariant local_solver,
          PSMF::LaplaceVariant     smooth_vmult,
          PSMF::SmootherVariant    smooth_inv>
void
LaplaceProblem<dim, fe_degree>::do_smooth()
{
  // DP
  using MatrixTypeDP =
    PSMF::LaplaceOperator<dim, fe_degree, full_number, smooth_vmult>;
  MatrixTypeDP matrix_dp;
  matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);

  using SmootherTypeDP = PSMF::PatchSmoother<MatrixTypeDP,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inv>;
  SmootherTypeDP                          smooth_dp;
  typename SmootherTypeDP::AdditionalData smoother_data_dp;
  smoother_data_dp.data         = mfdata_dp;
  smoother_data_dp.n_iterations = CT::N_SMOOTH_STEPS_;

  smooth_dp.initialize(matrix_dp, smoother_data_dp);

  auto assign_vector_cuda = [](auto &vec) {
    LinearAlgebra::ReadWriteVector<double> rw_vector(vec.size());
    for (unsigned int i = 0; i < rw_vector.size(); ++i)
      rw_vector(i) = i;

    vec.import(rw_vector, VectorOperation::insert);
  };

  Timer  time;
  double best_time = 1e10;

  assign_vector_cuda(system_rhs_dp);
  assign_vector_cuda(solution_dp);

  solution_dp   = 0;
  system_rhs_dp = 1;

  // system_rhs_dp.print(std::cout);

  // smooth_dp.step(solution_dp, system_rhs_dp);

  // std::cout << "TESTING SMOOTHER!!!\n";
  // solution_dp.print(std::cout);
  // std::cout << "\nTESTING SMOOTHER!!!\n";

  std::srand(0);
  LinearAlgebra::ReadWriteVector<full_number> rw_vector(dof_handler.n_dofs());
  for (unsigned int i = 0; i < rw_vector.size(); ++i)
    rw_vector[i] = ((double)std::rand()) / RAND_MAX;
  system_rhs_dp.import(rw_vector, VectorOperation::insert);

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          smooth_dp.step(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  std::cout << std::setprecision(17) << solution_dp.l2_norm() << std::endl;

  info_table[3].add_value("Name",
                          std::string(LaplaceToString(smooth_vmult)) + " " +
                            std::string(SmootherToString(smooth_inv)) + " DP");
  info_table[3].add_value("Time[s]", best_time);
  info_table[3].add_value("Perf[Dof/s]", n_dofs / best_time);

  // SP
  using MatrixTypeSP =
    PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, smooth_vmult>;
  MatrixTypeSP matrix_sp;
  matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);

  using SmootherTypeSP = PSMF::PatchSmoother<MatrixTypeSP,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inv>;
  SmootherTypeSP                          smooth_sp;
  typename SmootherTypeSP::AdditionalData smoother_data_sp;
  smoother_data_sp.data         = mfdata_sp;
  smoother_data_sp.n_iterations = CT::N_SMOOTH_STEPS_;

  smooth_sp.initialize(matrix_sp, smoother_data_sp);

  system_rhs_sp = 1.;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          smooth_sp.step(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  std::cout << std::setprecision(17) << solution_sp.l2_norm() << std::endl;

  info_table[4].add_value("Name",
                          std::string(LaplaceToString(smooth_vmult)) + " " +
                            std::string(SmootherToString(smooth_inv)) + " SP");
  info_table[4].add_value("Time[s]", best_time);
  info_table[4].add_value("Perf[Dof/s]", n_dofs / best_time);

  *pcout << "Smoother Mem: " << disp_gpu_usage() << " MB" << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_smooth()
{
  *pcout << "Benchmarking Smoothing...\n";

  do_smooth<CT::LOCAL_SOLVER_[0], CT::SMOOTH_VMULT_[0], CT::SMOOTH_INV_[0]>();

  // for (unsigned int k = 0; k < CT::SMOOTH_VMULT_.size(); ++k)
  //   switch (CT::SMOOTH_VMULT_[k])
  //     {
  //       case PSMF::LaplaceVariant::Basic:
  //         for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
  //           switch (CT::SMOOTH_INV_[j])
  //             {
  //               case PSMF::SmootherVariant::GLOBAL:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::Basic,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::Basic,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                     }
  //                 break;
  //               case PSMF::SmootherVariant::ConflictFree:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::Basic,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::Basic,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                     }
  //                 break;
  //             }
  //         break;
  //       case PSMF::LaplaceVariant::MatrixStruct:
  //         for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
  //           switch (CT::SMOOTH_INV_[j])
  //             {
  //               case PSMF::SmootherVariant::GLOBAL:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::MatrixStruct,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::MatrixStruct,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                     }
  //                 break;
  //               case PSMF::SmootherVariant::ConflictFree:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                     }
  //                 break;
  //             }
  //         break;
  //       case PSMF::LaplaceVariant::ConflictFree:
  //         for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
  //           switch (CT::SMOOTH_INV_[j])
  //             {
  //               case PSMF::SmootherVariant::GLOBAL:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::GLOBAL>();
  //                         break;
  //                     }
  //                 break;
  //               case PSMF::SmootherVariant::ConflictFree:
  //                 for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
  //                   switch (CT::LOCAL_SOLVER_[k])
  //                     {
  //                       case PSMF::LocalSolverVariant::Direct:
  //                         do_smooth<PSMF::LocalSolverVariant::Direct,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                       case PSMF::LocalSolverVariant::Bila:
  //                       case PSMF::LocalSolverVariant::KSVD:
  //                         do_smooth<PSMF::LocalSolverVariant::KSVD,
  //                                   PSMF::LaplaceVariant::ConflictFree,
  //                                   PSMF::SmootherVariant::ConflictFree>();
  //                         break;
  //                     }
  //                 break;
  //             }
  //         break;
  //       default:
  //         AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
  //     }
}
template <int dim, int fe_degree>
size_t
LaplaceProblem<dim, fe_degree>::disp_gpu_usage()
{
  size_t free_mem, total_mem;
  AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

  unsigned int scale    = 1024 * 1024;
  size_t       used_mem = (total_mem - free_mem) / scale;

  return used_mem;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  *pcout << Util::generic_info_to_fstring() << std::endl;

  GridGenerator::hyper_cube(triangulation, 0., 1.);

  double n_dofs_1d = 0;
  if (dim == 2)
    n_dofs_1d = std::sqrt(CT::MAX_SIZES_ / (dim + 1));
  else if (dim == 3)
    n_dofs_1d = std::cbrt(CT::MAX_SIZES_ / (dim + 1));

  auto n_refinement =
    static_cast<unsigned int>(std::log2(n_dofs_1d / (fe_degree + 1)));

#if MODE == 0
  triangulation.refine_global(n_refinement);
  setup_system();
  // assemble_rhs();
  bench_Ax();
  bench_transfer();
  bench_smooth();
#elif MODE == 1
  for (unsigned int cycle = 0; cycle < n_refinement; ++cycle)
    {
      triangulation.refine_global(1);
      setup_system();
      // assemble_rhs();
      bench_Ax();
      bench_transfer();
      bench_smooth();
    }
#endif

  *pcout << std::endl;

  for (unsigned int k = 0; k < 5; ++k)
    {
      std::ostringstream oss;

      info_table[k].set_scientific("Time[s]", true);
      info_table[k].set_precision("Time[s]", 3);
      info_table[k].set_scientific("Perf[Dof/s]", true);
      info_table[k].set_precision("Perf[Dof/s]", 3);

      info_table[k].write_text(oss);
      *pcout << oss.str() << std::endl;
      *pcout << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      {
        int device_id = findCudaDevice(argc, (const char **)argv);
        AssertCuda(cudaSetDevice(device_id));
      }

      {
        LaplaceProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
