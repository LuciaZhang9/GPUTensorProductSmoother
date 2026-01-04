#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iomanip>

#include "ct_parameter.h"
#include "patch_base.cuh"

using namespace dealii;

template <int dim, int degree>
void
run()
{
  using MatrixFree = PSMF::LevelVertexPatch<dim, degree, double>;

  typename MatrixFree::AdditionalData additional_data;
  additional_data.relaxation         = 1.;
  additional_data.use_coloring       = false;
  additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
  additional_data.granularity_scheme = CT::GRANULARITY_;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(7);

  DoFHandler<dim>   dof_handler(triangulation);
  MGConstrainedDoFs mg_constrained_dofs;
  FE_Q<dim>         fe(degree);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  for (unsigned int level = 1; level < triangulation.n_global_levels(); ++level)
    {
      MatrixFree mfdata;
      mfdata.reinit(dof_handler, mg_constrained_dofs, level, additional_data);

      auto mass_tensor      = mfdata.assemble_mass_tensor().back();
      auto laplace_tensor   = mfdata.assemble_laplace_tensor().back();
      auto bilaplace_tensor = mfdata.assemble_bilaplace_tensor();

      auto print_matrices = [&](std::string filename, auto matrix, auto pos) {
        filename += std::to_string(dim);
        filename += "D_Q";
        filename += std::to_string(degree);
        filename += "_L";
        filename += std::to_string(level);
        filename += "_pos";
        filename += std::to_string(pos);
        filename += ".txt";

        std::ofstream out;
        out.open(filename);

        for (unsigned int i = 0; i < 2 * degree + 1; ++i)
          {
            for (unsigned int j = 0; j < 2 * degree + 1; ++j)
              out << matrix(i, j) << " ";
            out << std::endl;
          }

        out.close();
      };

      print_matrices("mass_", mass_tensor, 0);
      print_matrices("laplace_", laplace_tensor, 0);
      for (unsigned int i = 0; i < 3; ++i)
        print_matrices("bilaplace_", bilaplace_tensor[i + 3], i);
    }
}

void
read_nn()
{
  std::string filename =
    "/export/home/cucui/CLionProjects/python-project-template/biharm/a0_interior.txt";

  std::ifstream file(filename);

  constexpr unsigned int N = 49;

  double a0[N];

  if (file.is_open())
    {
      std::istream_iterator<double> fileIter(file);
      std::copy_n(fileIter, N, a0);
      file.close();
    }
  else
    {
      std::cout << "Error opening file!" << std::endl;
    }

  for (unsigned int i = 0; i < N; i++)
    {
      std::cout << std::setprecision(10) << a0[i] << " ";
    }
  std::cout << std::endl;
}

int
main()
{
  // run<2, 2>();
  // run<2, 3>();
  // run<2, 4>();
  // run<2, 5>();
  // run<2, 6>();
  run<2, 7>();

  // read_nn();

  return 0;
}