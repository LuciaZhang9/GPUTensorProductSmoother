/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef TENSOR_PRODUCT_CUH
#define TENSOR_PRODUCT_CUH

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/tensor_product_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

namespace PSMF
{

  using namespace dealii;

  template <int dim, int fe_degree, typename Number, int n_rows_1d = -1>
  class TensorProductData
    : public TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>
  {
  public:
    void
    generate_1d_mf(unsigned int n_refinement);

    template <
      bool interior,
      typename CellIndices = std::array<std::array<unsigned int, 2>, dim>,
      bool inverse         = true>
    void
    compute_tensor_product(const CellIndices &cell_indices);

    void
    get_mass_matrix(std::array<Table<2, Number>, dim> &mass_matrix) const;

    void
    get_derivative_matrix(
      std::array<Table<2, Number>, dim> &derivative_matrix) const;

    void
    get_eigenvalues(std::array<AlignedVector<Number>, dim> &eigenvalues) const;

    void
    get_eigenvectors(std::array<Table<2, Number>, dim> &eigenvectors) const;

  private:
    MatrixFree<1, Number> matrix_free_1d;
  };

  // *************** implementation ***************
  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::generate_1d_mf(
    unsigned int n_refinement)
  {
    Triangulation<1> tria(Triangulation<1>::limit_level_difference_at_vertices);
    GridGenerator::hyper_cube(tria, 0, 1);
    tria.refine_global(n_refinement);

    FE_Q<1>       fe(fe_degree);
    DoFHandler<1> dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    AffineConstraints<Number> constraint;
    MappingQ<1>               mapping(1);

    constraint.clear();
    VectorTools::interpolate_boundary_values(
      mapping,
      dof_handler,
      0,
      Functions::ZeroFunction<1, Number>(),
      constraint);
    VectorTools::interpolate_boundary_values(
      mapping,
      dof_handler,
      1,
      Functions::ZeroFunction<1, Number>(),
      constraint);
    constraint.close();

    QGauss<1> quad(fe_degree + 1);

    typename MatrixFree<1, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<1, Number>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points);
    additional_data.hold_all_faces_to_owned_cells = true;
    additional_data.mg_level                      = tria.n_global_levels() - 1;

    matrix_free_1d.reinit(
      mapping, dof_handler, constraint, quad, additional_data);
  }
  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  template <bool interior, typename CellIndices, bool inverse>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::compute_tensor_product(
    const CellIndices &cell_indices)
  {
    static constexpr unsigned int dofs_per_cell = fe_degree + 1;
    static constexpr unsigned int stride        = fe_degree;

    FEEvaluation<1, fe_degree, fe_degree + 1, 1, Number> fe_eval_fun(
      matrix_free_1d);
    FEEvaluation<1, fe_degree, fe_degree + 1, 1, Number> fe_eval_der(
      matrix_free_1d);

    std::array<FullMatrix<Number>, dim> mass;
    std::array<FullMatrix<Number>, dim> derivative;
    std::array<FullMatrix<Number>, dim> mass_all;
    std::array<FullMatrix<Number>, dim> derivative_all;

    for (unsigned d = 0; d < dim; ++d)
      {
        if (interior)
          {
            mass[d]       = FullMatrix<Number>(2 * fe_degree - 1);
            derivative[d] = FullMatrix<Number>(2 * fe_degree - 1);
          }
        else
          {
            mass_all[d]       = FullMatrix<Number>(2 * fe_degree + 1);
            derivative_all[d] = FullMatrix<Number>(2 * fe_degree + 1);
          }
      }
    for (unsigned d = 0; d < dim; ++d)
      {
        auto        &dir    = cell_indices[d];
        unsigned int offset = 0;
        unsigned int shift  = 0;
        for (auto &cell : dir)
          {
            fe_eval_fun.reinit(cell);
            fe_eval_der.reinit(cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Set unit vectors
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    fe_eval_fun.begin_dof_values()[j] = 0.;
                    fe_eval_der.begin_dof_values()[j] = 0.;
                  }
                fe_eval_fun.begin_dof_values()[i] = 1.;
                fe_eval_der.begin_dof_values()[i] = 1.;

                // Apply operator on unit vector to generate the next few matrix
                // columns
                fe_eval_fun.evaluate(EvaluationFlags::values);
                fe_eval_der.evaluate(EvaluationFlags::gradients);
                for (unsigned int q = 0; q < fe_eval_fun.n_q_points; ++q)
                  {
                    fe_eval_fun.submit_value(1. * fe_eval_fun.get_value(q), q);
                    fe_eval_der.submit_gradient(fe_eval_der.get_gradient(q), q);
                  }
                fe_eval_fun.integrate(EvaluationFlags::values);
                fe_eval_der.integrate(EvaluationFlags::gradients);
                // Insert computed entries in matrix
                if (interior)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        if (i + offset == 0 || i + offset == 2 * fe_degree)
                          {
                            shift++;
                            break;
                          }

                        if (j + offset == 0 || j + offset == 2 * fe_degree)
                          {
                            continue;
                          }
                        mass[d](j + offset - shift, i + offset - shift) +=
                          fe_eval_fun.get_dof_value(j)[0];
                        derivative[d](j + offset - shift, i + offset - shift) +=
                          fe_eval_der.get_dof_value(j)[0];
                      }
                  }
                else
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        mass_all[d](j + offset, i + offset) +=
                          fe_eval_fun.get_dof_value(j)[0];
                        derivative_all[d](j + offset, i + offset) +=
                          fe_eval_der.get_dof_value(j)[0];
                      }
                  }
              }
            offset += stride;
          }
      }
    if (inverse)
      {
        if (interior)
          this->reinit(mass, derivative);
        else
          this->reinit(mass_all, derivative_all);
      }
    else
      {
        std::array<Table<2, Number>, dim> mass_copy;
        std::array<Table<2, Number>, dim> deriv_copy;

        std::transform(mass_all.cbegin(),
                       mass_all.cend(),
                       mass_copy.begin(),
                       [](const FullMatrix<Number> &m) -> Table<2, Number> {
                         return m;
                       });
        std::transform(derivative_all.cbegin(),
                       derivative_all.cend(),
                       deriv_copy.begin(),
                       [](const FullMatrix<Number> &m) -> Table<2, Number> {
                         return m;
                       });
        this->mass_matrix       = mass_copy;
        this->derivative_matrix = deriv_copy;
      }
  }

  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::get_mass_matrix(
    std::array<Table<2, Number>, dim> &mass_matrix) const
  {
    mass_matrix = this->mass_matrix;
  }
  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::get_derivative_matrix(
    std::array<Table<2, Number>, dim> &derivative_matrix) const
  {
    derivative_matrix = this->derivative_matrix;
  }
  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::get_eigenvalues(
    std::array<AlignedVector<Number>, dim> &eigenvalues) const
  {
    eigenvalues = this->eigenvalues;
  }
  template <int dim, int fe_degree, typename Number, int n_rows_1d>
  void
  TensorProductData<dim, fe_degree, Number, n_rows_1d>::get_eigenvectors(
    std::array<Table<2, Number>, dim> &eigenvectors) const
  {
    eigenvectors = this->eigenvectors;
  }
} // namespace PSMF

#endif // TENSOR_PRODUCT_CUH