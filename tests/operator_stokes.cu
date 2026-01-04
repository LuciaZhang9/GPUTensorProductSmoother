/**
 * Created by Cu Cui on 2023/4/17.
 */

// Testing stokes operator

#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/matrix_free/shape_info.h>

#include <iostream>

#include "TPSS/tensor_product_matrix.h"
#include "tensor_product.h"

using namespace dealii;

template <int dim, int degree, typename Number = double>
void
test()
{
  FE_RaviartThomas_new<dim> fe(degree);
  QGauss<1>                 quadrature(degree + 2);

  const unsigned int n_quadrature    = quadrature.size();
  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();

  std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

  for (unsigned int d = 0; d < dim; ++d)
    {
      n_cell_dofs_1d[d] = d == 0 ? degree + 2 : degree + 1;
      n_patch_dofs_1d[d] =
        d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
    }
  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
  shape_info.reinit(quadrature, fe);

  std::array<internal::MatrixFreeFunctions::UnivariateShapeData<Number>, dim>
    shape_data;
  for (auto d = 0U; d < dim; ++d)
    shape_data[d] = shape_info.get_shape_data(d, 0);

  const Number h              = 2.0;
  const Number penalty_factor = 1 * h * (degree + 1) * (degree + 2);

  auto cell_mass = [&](unsigned int pos) {
    std::array<Table<2, Number>, dim> mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mass_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

    unsigned int is_first = pos == 0 ? 1 : 0;

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            Number sum_mass = 0;
            for (unsigned int q = 0; q < n_quadrature; ++q)
              {
                sum_mass += shape_data[d].shape_values[i * n_quadrature + q] *
                            shape_data[d].shape_values[j * n_quadrature + q] *
                            quadrature.weight(q) * is_first;
              }

            mass_matrices[d](i, j) += sum_mass;
          }

    return mass_matrices;
  };


  auto cell_laplace = [&](unsigned int type, unsigned int pos) {
    std::array<Table<2, Number>, dim> laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      laplace_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

    unsigned int is_first = pos == 0 ? 1 : 0;

    Number boundary_factor_left  = 1.;
    Number boundary_factor_right = 1.;

    if (type == 0)
      boundary_factor_left = 2.;
    else if (type == 1 && pos == 0)
      boundary_factor_left = 0.;
    else if (type == 1 && pos == 1)
      boundary_factor_right = 0.;
    else if (type == 2)
      boundary_factor_right = 2.;
    else if (type == 3)
      is_first = 1;

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            Number sum_laplace = 0;
            for (unsigned int q = 0; q < n_quadrature; ++q)
              {
                sum_laplace +=
                  shape_data[d].shape_gradients[i * n_quadrature + q] *
                  shape_data[d].shape_gradients[j * n_quadrature + q] *
                  quadrature.weight(q) * h * h * is_first;
              }

            // bd
            if (d != 0)
              {
                sum_laplace +=
                  boundary_factor_left *
                  (1. * shape_data[d].shape_data_on_face[0][i] *
                     shape_data[d].shape_data_on_face[0][j] * penalty_factor +
                   0.5 *
                     shape_data[d]
                       .shape_data_on_face[0][i + n_cell_dofs_1d[d]] *
                     shape_data[d].shape_data_on_face[0][j] * h +
                   0.5 *
                     shape_data[d]
                       .shape_data_on_face[0][j + n_cell_dofs_1d[d]] *
                     shape_data[d].shape_data_on_face[0][i] * h) *
                  h;

                sum_laplace +=
                  boundary_factor_right *
                  (1. * shape_data[d].shape_data_on_face[1][i] *
                     shape_data[d].shape_data_on_face[1][j] * penalty_factor -
                   0.5 *
                     shape_data[d]
                       .shape_data_on_face[1][i + n_cell_dofs_1d[d]] *
                     shape_data[d].shape_data_on_face[1][j] * h -
                   0.5 *
                     shape_data[d]
                       .shape_data_on_face[1][j + n_cell_dofs_1d[d]] *
                     shape_data[d].shape_data_on_face[1][i] * h) *
                  h;
              }

            laplace_matrices[d](i, j) += sum_laplace;
          }

    return laplace_matrices;
  };

  auto cell_mixed = [&]() {
    std::array<Table<2, Number>, dim> mixed_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mixed_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            if (d != 0)
              {
                mixed_matrices[d](j, i) =
                  -0.5 *
                    shape_data[d].shape_data_on_face[0][i + n_cell_dofs_1d[d]] *
                    shape_data[d].shape_data_on_face[1][j] * h * h +
                  0.5 * shape_data[d].shape_data_on_face[0][i] *
                    shape_data[d].shape_data_on_face[1][j + n_cell_dofs_1d[d]] *
                    h * h;
              }
          }

    return mixed_matrices;
  };

  auto cell_penalty = [&]() {
    std::array<Table<2, Number>, dim> penalty_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      penalty_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            if (d != 0)
              {
                penalty_matrices[d](j, i) =
                  -1. * shape_data[d].shape_data_on_face[0][i] *
                  shape_data[d].shape_data_on_face[1][j] * penalty_factor * h;
              }
          }

    return penalty_matrices;
  };

  auto print_matrices = [](auto matrix) {
    for (auto i = 0U; i < matrix.size(); ++i)
      {
        for (auto m = 0U; m < matrix[i].size(0); ++m)
          {
            for (auto n = 0U; n < matrix[i].size(1); ++n)
              std::cout << matrix[i](m, n) << " ";
            std::cout << std::endl;
          }
        std::cout << std::endl;
      }
    std::cout << std::endl;
  };

  auto mass_matrices0    = cell_mass(0);
  auto mass_matrices1    = cell_mass(1);
  auto laplace_matrices0 = cell_laplace(0, 0);
  auto laplace_matrices1 = cell_laplace(2, 0);

  auto mixed   = cell_mixed();
  auto penalty = cell_penalty();

  // print_matrices(mixed);
  // print_matrices(penalty);

  auto patch_mass = [&](auto left, auto right) {
    std::array<Table<2, Number>, dim> mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mass_matrices[d].reinit(n_patch_dofs_1d[d], n_patch_dofs_1d[d]);

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            unsigned int shift = d == 0;
            mass_matrices[d](i, j) += left[d](i, j);
            mass_matrices[d](i + n_cell_dofs_1d[d] - shift,
                             j + n_cell_dofs_1d[d] - shift) += right[d](i, j);
          }
    return mass_matrices;
  };

  auto patch_laplace = [&](auto left, auto right) {
    std::array<Table<2, Number>, dim> laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      laplace_matrices[d].reinit(n_patch_dofs_1d[d], n_patch_dofs_1d[d]);

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            unsigned int shift = d == 0;
            laplace_matrices[d](i, j) += left[d](i, j);
            laplace_matrices[d](i + n_cell_dofs_1d[d] - shift,
                                j + n_cell_dofs_1d[d] - shift) +=
              right[d](i, j);

            laplace_matrices[d](i, j + n_cell_dofs_1d[d] - shift) +=
              mixed[d](i, j);
            laplace_matrices[d](i, j + n_cell_dofs_1d[d] - shift) +=
              penalty[d](i, j);

            if (d != 0)
              {
                laplace_matrices[d](j + n_cell_dofs_1d[d] - shift, i) =
                  laplace_matrices[d](i, j + n_cell_dofs_1d[d] - shift);
              }
          }
    return laplace_matrices;
  };

  auto mass    = patch_mass(mass_matrices0, mass_matrices0);
  auto laplace = patch_laplace(laplace_matrices0, laplace_matrices1);

  print_matrices(mass);
  print_matrices(laplace);

  PSMF::TensorProductData<dim, degree, Number> tensor_product;
  tensor_product.reinit(mass, laplace);

  std::array<AlignedVector<Number>, dim> eigenvalue_tensor;
  std::array<Table<2, Number>, dim>      eigenvector_tensor;
  tensor_product.get_eigenvalues(eigenvalue_tensor);
  tensor_product.get_eigenvectors(eigenvector_tensor);

  for (auto eigs : eigenvalue_tensor)
    {
      for (auto e : eigs)
        std::cout << e << " ";
      std::cout << std::endl;
    }

  std::cout << std::endl;
  print_matrices(eigenvector_tensor);

  // kron(L2,M1)+kron(M2, L1)
}

template <int dim, int degree, typename Number = double>
void
test_mixed()
{
  FE_RaviartThomas_new<dim> fe_v(degree);
  FE_DGQLegendre<dim>       fe_p(degree);
  QGauss<1>                 quadrature(degree + 2);

  const unsigned int n_quadrature    = quadrature.size();
  const unsigned int n_dofs_per_cell = fe_v.n_dofs_per_cell();

  std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

  for (unsigned int d = 0; d < dim; ++d)
    {
      n_cell_dofs_1d[d] = d == 0 ? degree + 2 : degree + 1;
      n_patch_dofs_1d[d] =
        d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
    }
  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_v;
  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_p;
  shape_info_v.reinit(quadrature, fe_v);
  shape_info_p.reinit(quadrature, fe_p);

  std::array<internal::MatrixFreeFunctions::UnivariateShapeData<Number>, dim>
    shape_data_v;
  std::array<internal::MatrixFreeFunctions::UnivariateShapeData<Number>, dim>
    shape_data_p;
  for (auto d = 0U; d < dim; ++d)
    {
      shape_data_v[d] = shape_info_v.get_shape_data(d, 0);
      shape_data_p[d] = shape_info_p.get_shape_data(d, 0);
    }

  const Number h = 2.0;

  auto cell_mass = [&](unsigned int pos) {
    std::array<Table<2, Number>, dim> mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mass_matrices[d].reinit(n_cell_dofs_1d[d], degree + 1);

    unsigned int is_first = pos == 0 ? 1 : 0;

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          {
            Number sum_mass = 0;
            for (unsigned int q = 0; q < n_quadrature; ++q)
              {
                sum_mass += shape_data_v[d].shape_values[i * n_quadrature + q] *
                            shape_data_p[d].shape_values[j * n_quadrature + q] *
                            quadrature.weight(q) * is_first;
              }

            mass_matrices[d](i, j) += sum_mass;
          }

    return mass_matrices;
  };


  auto cell_laplace = [&](unsigned int pos) {
    std::array<Table<2, Number>, dim> laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      laplace_matrices[d].reinit(n_cell_dofs_1d[d], degree + 1);

    unsigned int is_first = pos == 0 ? 1 : 0;

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          {
            Number sum_laplace = 0;
            for (unsigned int q = 0; q < n_quadrature; ++q)
              {
                sum_laplace +=
                  shape_data_v[d].shape_gradients[i * n_quadrature + q] *
                  shape_data_p[d].shape_values[j * n_quadrature + q] *
                  quadrature.weight(q) * is_first;
              }

            laplace_matrices[d](i, j) += sum_laplace;
          }

    return laplace_matrices;
  };

  auto print_matrices = [](auto matrix) {
    for (auto i = 0U; i < matrix.size(); ++i)
      {
        for (auto m = 0U; m < matrix[i].size(0); ++m)
          {
            for (auto n = 0U; n < matrix[i].size(1); ++n)
              std::cout << matrix[i](m, n) << " ";
            std::cout << std::endl;
          }
        std::cout << std::endl;
      }
    std::cout << std::endl;
  };

  auto mass_matrices     = cell_mass(0);
  auto laplace_matrices0 = cell_laplace(0);
  auto laplace_matrices1 = cell_laplace(0);

  // print_matrices(mass_matrices);
  // print_matrices(laplace_matrices1);

  auto patch_mass = [&](auto left, auto right) {
    std::array<Table<2, Number>, dim> mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mass_matrices[d].reinit(n_patch_dofs_1d[d], 2 * degree + 2);

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          {
            unsigned int shift = d == 0;
            mass_matrices[d](i, j) += left[d](i, j);
            mass_matrices[d](i + n_cell_dofs_1d[d] - shift, j + degree + 1) +=
              right[d](i, j);
          }
    return mass_matrices;
  };

  auto patch_laplace = [&](auto left, auto right) {
    std::array<Table<2, Number>, dim> laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      laplace_matrices[d].reinit(n_patch_dofs_1d[d], 2 * degree + 2);

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          {
            unsigned int shift = d == 0;
            laplace_matrices[d](i, j) += left[d](i, j);
            laplace_matrices[d](i + n_cell_dofs_1d[d] - shift,
                                j + degree + 1) += right[d](i, j);
          }
    return laplace_matrices;
  };

  auto mass    = patch_mass(mass_matrices, mass_matrices);
  auto laplace = patch_laplace(laplace_matrices0, laplace_matrices1);

  print_matrices(mass);
  print_matrices(laplace);

  // -kron(M2,D1)
}

template <int dim, int degree, typename Number = double>
void
test_mass()
{
  FE_RaviartThomas_new<dim> fe_v(degree);
  FE_DGQLegendre<dim>       fe_p(degree);
  QGauss<1>                 quadrature(degree + 2);

  const unsigned int n_quadrature    = quadrature.size();
  const unsigned int n_dofs_per_cell = fe_v.n_dofs_per_cell();

  std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

  for (unsigned int d = 0; d < dim; ++d)
    {
      n_cell_dofs_1d[d] = d == 0 ? degree + 2 : degree + 1;
      n_patch_dofs_1d[d] =
        d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
    }
  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_v;
  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_p;
  shape_info_v.reinit(quadrature, fe_v);
  shape_info_p.reinit(quadrature, fe_p);

  std::array<internal::MatrixFreeFunctions::UnivariateShapeData<Number>, dim>
    shape_data_v;
  std::array<internal::MatrixFreeFunctions::UnivariateShapeData<Number>, dim>
    shape_data_p;
  for (auto d = 0U; d < dim; ++d)
    {
      shape_data_v[d] = shape_info_v.get_shape_data(d, 0);
      shape_data_p[d] = shape_info_p.get_shape_data(d, 0);
    }

  const Number h = 2.0;

  auto cell_mass = [&](unsigned int pos) {
    std::array<Table<2, Number>, dim> mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      mass_matrices[d].reinit(degree + 1, degree + 1);

    unsigned int is_first = pos == 0 ? 1 : 0;

    // dir0, mass & laplace
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < degree + 1; ++i)
        for (unsigned int j = 0; j < degree + 1; ++j)
          {
            Number sum_mass = 0;
            for (unsigned int q = 0; q < n_quadrature; ++q)
              {
                sum_mass += shape_data_p[d].shape_values[i * n_quadrature + q] *
                            shape_data_p[d].shape_values[j * n_quadrature + q] *
                            quadrature.weight(q) * is_first;
              }

            mass_matrices[d](i, j) += sum_mass;
          }

    return mass_matrices;
  };

  auto print_matrices = [](auto matrix) {
    for (auto i = 0U; i < matrix.size(); ++i)
      {
        for (auto m = 0U; m < matrix[i].size(0); ++m)
          {
            for (auto n = 0U; n < matrix[i].size(1); ++n)
              std::cout << matrix[i](m, n) << " ";
            std::cout << std::endl;
          }
        std::cout << std::endl;
      }
    std::cout << std::endl;
  };

  auto mass_matrices = cell_mass(0);

  print_matrices(mass_matrices);
}


int
main()
{
  // test<3, 2>();

  // test_mixed<3, 2>();

  // test_mixed<2, 6>();

  // test<2, 5>();

  test_mass<2, 4>();
}
