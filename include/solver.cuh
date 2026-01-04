/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of solvers
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <deal.II/base/function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <functional>
#include <iomanip>

#include "cuda_mg_transfer.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

using namespace dealii;

namespace PSMF
{
  // A coarse solver defined via the smoother
  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };

  struct SolverData
  {
    std::string solver_name = "";

    int    n_iteration      = 0;
    double n_step           = 0;
    double residual         = 0.;
    double reduction_rate   = 0.;
    double convergence_rate = 0.;
    double l2_error         = 0.;
    int    mem_usage        = 0.;
    double timing           = 0.;
    double perf             = 0.;

    std::string
    print_comp()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);
      oss.setf(std::ios::scientific);

      oss << std::left << std::setw(12) << solver_name << std::setprecision(4)
          << std::setw(12) << timing << std::setprecision(4) << std::setw(12)
          << perf << std::endl;

      return oss.str();
    }

    std::string
    print_solver()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);

      oss << std::left << std::setw(12) << solver_name << std::setw(4)
          << n_iteration << std::setw(8) << n_step;

      oss.setf(std::ios::scientific);

      // if (CT::SETS_ == "error_analysis")
      oss << std::setprecision(4) << std::setw(12) << timing << std::left
          << std::setprecision(4) << std::setw(12) << residual
          << std::setprecision(4) << std::setw(12) << reduction_rate
          << std::setprecision(4) << std::setw(12) << convergence_rate;

      oss << std::left << std::setw(8) << mem_usage << std::endl;

      return oss.str();
    }
  };

  /**
   * @brief Multigrid Method with vertex-patch smoother.
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number full number
   * @tparam smooth_kernel
   * @tparam Number1 vcycle number
   */
  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver,
            LaplaceVariant     lapalace_kernel,
            LaplaceVariant     smooth_vmult,
            SmootherVariant    smooth_inverse,
            typename Number2>
  class MultigridSolver
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorType2 =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;
    using MatrixType = LaplaceOperator<dim, fe_degree, Number, lapalace_kernel>;
    using MatrixType2 =
      LaplaceOperator<dim, fe_degree, Number2, lapalace_kernel>;
    using SmootherType       = PatchSmoother<MatrixType2,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inverse>;
    using SmootherTypeCoarse = PatchSmoother<MatrixType2,
                                             dim,
                                             fe_degree,
                                             LocalSolverVariant::Direct,
                                             smooth_vmult,
                                             smooth_inverse>;
    using MatrixFreeType     = LevelVertexPatch<dim, fe_degree, Number>;
    using MatrixFreeType2    = LevelVertexPatch<dim, fe_degree, Number2>;

    MultigridSolver(
      const DoFHandler<dim>                                 &dof_handler,
      const DoFHandler<dim>                                 &dof_handler_v,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>>  &mfdata_dp,
      const MGLevelObject<std::shared_ptr<MatrixFreeType2>> &mfdata,
      const MGTransferCUDA<dim, Number2>                    &transfer,
      const VectorType                                      &right_hand_side,
      std::shared_ptr<ConditionalOStream>                    pcout,
      const unsigned int                                     n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , n_cycles(n_cycles)
      , pcout(pcout)
    {
      AssertDimension(fe_degree + 1, dof_handler.get_fe().degree);

      matrix_dp.resize(minlevel, maxlevel);
      matrix.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix_dp[level].initialize(mfdata_dp[level], dof_handler, level);
          matrix[level].initialize(mfdata[level], dof_handler, level);

          matrix[level].initialize_dof_vector(defect[level]);
          solution_update[level].reinit(defect[level]);

          if (level == maxlevel)
            {
              matrix_dp[level].initialize_dof_vector(solution[level]);
              rhs[level].reinit(solution[level]);
            }
        }

      {
        // evaluate the right hand side in the equation, including the
        // residual from the inhomogeneous boundary conditions
        // set_inhomogeneous_bc<false>(maxlevel);
        rhs[maxlevel] = 0.;
        if (CT::SETS_ == "error_analysis")
          rhs[maxlevel] = right_hand_side;
        else
          rhs[maxlevel] = 1.;
      }


      {
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        MGLevelObject<typename SmootherTypeCoarse::AdditionalData>
          smoother_data_coarse;
        smoother_data.resize(minlevel, maxlevel);
        smoother_data_coarse.resize(minlevel, maxlevel);
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            smoother_data[level].data         = mfdata[level];
            smoother_data[level].n_iterations = CT::N_SMOOTH_STEPS_;

            smoother_data_coarse[level].data         = mfdata[level];
            smoother_data_coarse[level].n_iterations = CT::N_SMOOTH_STEPS_;
          }

        mg_smoother.initialize(matrix, smoother_data);
        mg_smoother_coarse.initialize(matrix, smoother_data_coarse);
        mg_coarse.initialize(mg_smoother_coarse);
      }
    }

    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 10 : 4;

      auto tester = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 5;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.restart();
            for (unsigned int i = 0; i < n_mv; ++i)
              kernel(this);
            best_time = std::min(time.wall_time() / n_mv, best_time);
          }

        SolverData data;
        data.solver_name = comp_name;
        data.timing      = best_time;
        data.perf        = n_dofs / best_time;
        comp_data.push_back(data);
      };

      for (unsigned int s = 0; s < 3; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_matvec);
                  comp_name   = "Mat-vec DP";
                  tester(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel =
                    std::mem_fn(&MultigridSolver::do_matvec_smoother);
                  comp_name = "Mat-vec SP";
                  tester(kernel);
                  break;
                }
              case 2:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_smoother);
                  comp_name   = "Smooth";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return comp_data;
    }

    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      all_mg_counter++;

      transfer->copy_to_mg(*dof_handler, defect, src);

      preconditioner_mg->vmult(solution_update[maxlevel], defect[maxlevel]);

      transfer->copy_from_mg(*dof_handler, dst, solution_update);
    }

    // Return the solution vector for further processing
    const VectorType &
    get_solution()
    {
      return solution[maxlevel];
    }

    void
    print_timings() const
    {
      // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        *pcout << " - #N of calls of multigrid: " << all_mg_counter << std::endl
               << std::endl;
        *pcout << " - Times of multigrid (levels):" << std::endl;

        const auto print_line = [&](const auto &vector) {
          for (const auto &i : vector)
            *pcout << std::scientific << std::setprecision(2) << std::setw(10)
                   << i.first;

          double sum = 0;

          for (const auto &i : vector)
            sum += i.first;

          *pcout << "   | " << std::scientific << std::setprecision(2)
                 << std::setw(10) << sum;

          *pcout << "\n";
        };

        for (unsigned int l = 0; l < all_mg_timers.size(); ++l)
          {
            *pcout << std::setw(4) << l << ": ";

            print_line(all_mg_timers[l]);
          }

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          sums(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size(); ++j)
            sums[i].first += all_mg_timers[j][i].first;

        *pcout
          << "   ----------------------------------------------------------------------------+-----------\n";
        *pcout << "      ";
        print_line(sums);

        *pcout << std::endl;

        *pcout << " - Times of multigrid (solver <-> mg): ";

        for (const auto &i : all_mg_precon_timers)
          *pcout << i.first << " ";
        *pcout << std::endl;
        *pcout << std::endl;
      }
    }

    void
    clear_timings() const
    {
      for (auto &is : all_mg_timers)
        for (auto &i : is)
          i.first = 0.0;

      for (auto &i : all_mg_precon_timers)
        i.first = 0.0;

      all_mg_counter = 0;
    }


    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per GMRES iteration
    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving...\n";

      std::string solver_name = "GMRES";

      mg::Matrix<VectorType2> mg_matrix(matrix);

      Multigrid<VectorType2> mg(mg_matrix,
                                mg_coarse,
                                *transfer,
                                mg_smoother,
                                mg_smoother,
                                minlevel,
                                maxlevel);

      preconditioner_mg = std::make_unique<
        PreconditionMG<dim, VectorType2, MGTransferCUDA<dim, Number2>>>(
        *dof_handler, mg, *transfer);

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverGMRES<VectorType> solver(solver_control);

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;
      for (unsigned int i = 0; i < N; ++i)
        {
          time.reset();
          time.start();

          solution[maxlevel] = 0;
          solver.solve(matrix_dp[maxlevel],
                       solution[maxlevel],
                       rhs[maxlevel],
                       *this);

          best_time = std::min(time.wall_time(), best_time);
        }

      auto n_iter     = solver_control.last_step();
      auto residual_0 = solver_control.initial_value();
      auto residual_n = solver_control.last_value();
      auto reduction  = solver_control.reduction();

      // *** average reduction: r_n = rho^n * r_0
      const double rho =
        std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
      const double convergence_rate =
        1. / n_iter * std::log10(residual_0 / residual_n);

      const auto n_step = -10 * std::log10(rho);
      const auto n_frac = std::log(reduction) / std::log(rho);

      size_t free_mem, total_mem;
      AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

      int mem_usage = (total_mem - free_mem) / 1024 / 1024;

      SolverData data;
      data.solver_name      = solver_name;
      data.n_iteration      = n_iter;
      data.n_step           = n_frac;
      data.residual         = residual_n;
      data.reduction_rate   = rho;
      data.convergence_rate = convergence_rate;
      data.timing           = best_time;
      data.mem_usage        = mem_usage;
      solver_data.push_back(data);

      auto history_data = solver_control.get_history_data();
      for (auto i = 1U; i < n_iter + 1; ++i)
        *pcout << "step " << i << ": " << history_data[i] / residual_0 << "\n";

      return solver_data;
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dp[maxlevel].vmult(solution[maxlevel], solution[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      matrix[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

    // run smoother in single precision
    void
    do_smoother()
    {
      (mg_smoother)
        .smooth(maxlevel, solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

  private:
    const SmartPointer<const DoFHandler<dim>>              dof_handler;
    const SmartPointer<const MGTransferCUDA<dim, Number2>> transfer;

    MGLevelObject<MatrixType>  matrix_dp;
    MGLevelObject<MatrixType2> matrix;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable MGLevelObject<VectorType> solution;

    /**
     * Original right hand side vector
     */
    mutable MGLevelObject<VectorType> rhs;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType2> defect;

    /**
     * Auxiliary vector for the solution update
     */
    mutable MGLevelObject<VectorType2> solution_update;

    // MGLevelObject<SmootherType> smooth;

    MGSmootherPrecondition<MatrixType2, SmootherType, VectorType2> mg_smoother;

    /**
     * The coarse solver
     */

    MGSmootherPrecondition<MatrixType2, SmootherTypeCoarse, VectorType2>
      mg_smoother_coarse;

    MGCoarseGridApplySmoother<VectorType2> mg_coarse;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<SolverData> solver_data;

    std::shared_ptr<ConditionalOStream> pcout;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType2, MGTransferCUDA<dim, Number2>>>
      preconditioner_mg;

    mutable unsigned int all_mg_counter = 0;

    mutable std::vector<std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers;

    mutable std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
      all_mg_precon_timers;
  };



  template <int dim,
            int fe_degree,
            typename Number,
            LocalSolverVariant local_solver,
            LaplaceVariant     lapalace_kernel,
            LaplaceVariant     smooth_vmult,
            SmootherVariant    smooth_inverse>
  class MultigridSolver<dim,
                        fe_degree,
                        Number,
                        local_solver,
                        lapalace_kernel,
                        smooth_vmult,
                        smooth_inverse,
                        Number>
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using MatrixType = LaplaceOperator<dim, fe_degree, Number, lapalace_kernel>;
    using SmootherType       = PatchSmoother<MatrixType,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inverse>;
    using SmootherTypeCoarse = PatchSmoother<MatrixType,
                                             dim,
                                             fe_degree,
                                             local_solver,
                                             smooth_vmult,
                                             smooth_inverse>;
    using MatrixFreeType     = LevelVertexPatch<dim, fe_degree, Number>;

    MultigridSolver(
      const DoFHandler<dim>                                &dof_handler,
      const DoFHandler<dim>                                &dof_handler_v,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>> &mfdata_dp,
      const MGLevelObject<std::shared_ptr<MatrixFreeType>> &,
      const MGTransferCUDA<dim, Number>  &transfer_dp,
      const VectorType                   &right_hand_side,
      std::shared_ptr<ConditionalOStream> pcout,
      const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , transfer(&transfer_dp)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , n_cycles(n_cycles)
      , pcout(pcout)
    {
      AssertDimension(fe_degree + 1, dof_handler_v.get_fe().degree);

      matrix.resize(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          matrix[level].initialize(mfdata_dp[level],
                                   dof_handler,
                                   dof_handler_v,
                                   level);
        }

      matrix[maxlevel].initialize_dof_vector(solution);

      if (CT::SETS_ == "error_analysis")
        rhs = right_hand_side;
      else
        rhs = 1.;


      {
        MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
        MGLevelObject<typename SmootherTypeCoarse::AdditionalData>
          smoother_data_coarse;
        smoother_data.resize(minlevel, maxlevel);
        smoother_data_coarse.resize(minlevel, maxlevel);
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            smoother_data[level].data         = mfdata_dp[level];
            smoother_data[level].n_iterations = CT::N_SMOOTH_STEPS_;

            smoother_data_coarse[level].data         = mfdata_dp[level];
            smoother_data_coarse[level].n_iterations = CT::N_SMOOTH_STEPS_;
          }

        mg_smoother.initialize(matrix, smoother_data);
        mg_smoother_coarse.initialize(matrix, smoother_data_coarse);
        mg_coarse.initialize(mg_smoother_coarse);
      }
    }

    std::vector<SolverData>
    static_comp()
    {
      *pcout << "Testing...\n";

      std::vector<SolverData> comp_data;

      std::string comp_name = "";

      const unsigned int n_dofs = dof_handler->n_dofs();
      const unsigned int n_mv   = n_dofs < 10000000 ? 10 : 4;

      auto tester = [&](auto kernel) {
        Timer              time;
        const unsigned int N         = 5;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.restart();
            for (unsigned int i = 0; i < n_mv; ++i)
              kernel(this);
            best_time = std::min(time.wall_time() / n_mv, best_time);
          }

        SolverData data;
        data.solver_name = comp_name;
        data.timing      = best_time;
        data.perf        = n_dofs / best_time;
        comp_data.push_back(data);
      };

      for (unsigned int s = 0; s < 1; ++s)
        {
          switch (s)
            {
              case 0:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_matvec);
                  comp_name   = "Mat-vec";
                  tester(kernel);
                  break;
                }
              case 1:
                {
                  auto kernel = std::mem_fn(&MultigridSolver::do_smooth);
                  comp_name   = "Smooth";
                  tester(kernel);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Solver Variant."));
            }
        }

      return comp_data;
    }


    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      all_mg_counter++;

      preconditioner_mg->vmult(dst, src);
    }

    // Return the solution vector for further processing
    const VectorType &
    get_solution()
    {
      return solution;
    }

    void
    print_timings() const
    {
      // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        *pcout << " - #N of calls of multigrid: " << all_mg_counter << std::endl
               << std::endl;
        *pcout << " - Times of multigrid (levels):" << std::endl;

        const auto print_line = [&](const auto &vector) {
          for (const auto &i : vector)
            *pcout << std::scientific << std::setprecision(2) << std::setw(10)
                   << i.first;

          double sum = 0;

          for (const auto &i : vector)
            sum += i.first;

          *pcout << "   | " << std::scientific << std::setprecision(2)
                 << std::setw(10) << sum;

          *pcout << "\n";
        };

        for (unsigned int l = 0; l < all_mg_timers.size(); ++l)
          {
            *pcout << std::setw(4) << l << ": ";

            print_line(all_mg_timers[l]);
          }

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          sums(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size(); ++j)
            sums[i].first += all_mg_timers[j][i].first;

        *pcout
          << "   ----------------------------------------------------------------------------+-----------\n";
        *pcout << "      ";
        print_line(sums);

        *pcout << std::endl;

        *pcout << " - Times of multigrid (solver <-> mg): ";

        for (const auto &i : all_mg_precon_timers)
          *pcout << i.first << " ";
        *pcout << std::endl;
        *pcout << std::endl;
      }
    }

    void
    clear_timings() const
    {
      for (auto &is : all_mg_timers)
        for (auto &i : is)
          i.first = 0.0;

      for (auto &i : all_mg_precon_timers)
        i.first = 0.0;

      all_mg_counter = 0;
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per GMRES iteration
    std::vector<SolverData>
    solve()
    {
      *pcout << "Solving...\n";

      mg::Matrix<VectorType> mg_matrix(matrix);

      Multigrid<VectorType> mg_obj(mg_matrix,
                                   mg_coarse,
                                   *transfer,
                                   mg_smoother,
                                   mg_smoother,
                                   minlevel,
                                   maxlevel);

      Multigrid<VectorType> mg(mg_matrix,
                               mg_coarse,
                               *transfer,
                               mg_smoother,
                               mg_smoother,
                               minlevel,
                               maxlevel);

      preconditioner_mg = std::make_unique<
        PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
        *dof_handler, mg, *transfer);

      // timers
      if (true)
        {
          all_mg_timers.resize((maxlevel - minlevel + 1));
          for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
            all_mg_timers[i].resize(7);

          const auto create_mg_timer_function = [&](const unsigned int i,
                                                    const std::string &label) {
            return [i, label, this](const bool flag, const unsigned int level) {
              if (false && flag)
                std::cout << label << " " << level << std::endl;
              if (flag)
                all_mg_timers[level - minlevel][i].second =
                  std::chrono::system_clock::now();
              else
                all_mg_timers[level - minlevel][i].first +=
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() -
                    all_mg_timers[level - minlevel][i].second)
                    .count() /
                  1e9;
            };
          };

          {
            mg.connect_pre_smoother_step(
              create_mg_timer_function(0, "pre_smoother_step"));
            mg.connect_residual_step(
              create_mg_timer_function(1, "residual_step"));
            mg.connect_restriction(create_mg_timer_function(2, "restriction"));
            mg.connect_coarse_solve(
              create_mg_timer_function(3, "coarse_solve"));
            mg.connect_prolongation(
              create_mg_timer_function(4, "prolongation"));
            mg.connect_edge_prolongation(
              create_mg_timer_function(5, "edge_prolongation"));
            mg.connect_post_smoother_step(
              create_mg_timer_function(6, "post_smoother_step"));
          }

          all_mg_precon_timers.resize(2);

          const auto create_mg_precon_timer_function =
            [&](const unsigned int i) {
              return [i, this](const bool flag) {
                if (flag)
                  all_mg_precon_timers[i].second =
                    std::chrono::system_clock::now();
                else
                  all_mg_precon_timers[i].first +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now() -
                      all_mg_precon_timers[i].second)
                      .count() /
                    1e9;
              };
            };

          preconditioner_mg->connect_transfer_to_mg(
            create_mg_precon_timer_function(0));
          preconditioner_mg->connect_transfer_to_global(
            create_mg_precon_timer_function(1));
        }

      std::string solver_name = "GMRES";

      ReductionControl solver_control(CT::MAX_STEPS_, 1e-14, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverGMRES<VectorType> solver(solver_control);

      {
        solution = 0;
        solver.solve(matrix[maxlevel], solution, rhs, *this);
        print_timings();
        clear_timings();
      }

      preconditioner_mg = std::make_unique<
        PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
        *dof_handler, mg_obj, *transfer);

      Timer              time;
      const unsigned int N         = 5;
      double             best_time = 1e10;
      for (unsigned int i = 0; i < N; ++i)
        {
          time.reset();
          time.start();

          solution = 0;
          solver.solve(matrix[maxlevel], solution, rhs, *this);

          best_time = std::min(time.wall_time(), best_time);
        }

      auto n_iter     = solver_control.last_step();
      auto residual_0 = solver_control.initial_value();
      auto residual_n = solver_control.last_value();
      auto reduction  = solver_control.reduction();

      // *** average reduction: r_n = rho^n * r_0
      const double rho =
        std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
      const double convergence_rate =
        1. / n_iter * std::log10(residual_0 / residual_n);

      const auto n_step = -10 * std::log10(rho);
      const auto n_frac = std::log(reduction) / std::log(rho);

      size_t free_mem, total_mem;
      AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

      int mem_usage = (total_mem - free_mem) / 1024 / 1024;

      SolverData data;
      data.solver_name      = solver_name;
      data.n_iteration      = n_iter;
      data.n_step           = n_frac;
      data.residual         = residual_n;
      data.reduction_rate   = rho;
      data.convergence_rate = convergence_rate;
      data.timing           = best_time;
      data.mem_usage        = mem_usage;
      solver_data.push_back(data);

      auto history_data = solver_control.get_history_data();
      for (auto i = 1U; i < n_iter + 1; ++i)
        *pcout << "step " << i << ": " << history_data[i] / residual_0 << "\n";

      return solver_data;
    }

    // run smooth in double precision
    void
    do_smooth()
    {
      (mg_smoother).smooth(maxlevel, solution, rhs);
      cudaDeviceSynchronize();
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix[maxlevel].vmult(solution, rhs);
      cudaDeviceSynchronize();
    }

  private:
    const SmartPointer<const DoFHandler<dim>>             dof_handler;
    const SmartPointer<const MGTransferCUDA<dim, Number>> transfer;

    MGLevelObject<MatrixType> matrix;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable VectorType solution;

    /**
     * Original right hand side vector
     */
    mutable VectorType rhs;


    // MGLevelObject<SmootherType> smooth;

    MGSmootherPrecondition<MatrixType, SmootherType, VectorType> mg_smoother;

    /**
     * The coarse solver
     */
    MGSmootherPrecondition<MatrixType, SmootherTypeCoarse, VectorType>
      mg_smoother_coarse;

    MGCoarseGridApplySmoother<VectorType> mg_coarse;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<SolverData> solver_data;

    std::shared_ptr<ConditionalOStream> pcout;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>
      preconditioner_mg;

    mutable unsigned int all_mg_counter = 0;

    mutable std::vector<std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers;

    mutable std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
      all_mg_precon_timers;
  };

} // namespace PSMF

#endif // SOLVER_CUH
