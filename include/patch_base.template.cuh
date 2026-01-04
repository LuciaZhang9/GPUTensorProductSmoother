/**
 * @file patch_base.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <deal.II/base/graph_coloring.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/matrix_free/shape_info.h>

#include <omp.h>

#include <fstream>

#include "TPSS/tensors.h"
#include "loop_kernel.cuh"
#include "renumber.h"

namespace PSMF
{

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::LevelVertexPatch()
  {}

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::~LevelVertexPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::free()
  {
    for (auto &first_dof_color_ptr : first_dof_laplace)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_laplace.clear();

    for (auto &first_dof_color_ptr : first_dof_smooth)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_smooth.clear();

    for (auto &patch_id_color_ptr : patch_id)
      Utilities::CUDA::free(patch_id_color_ptr);
    patch_id.clear();

    for (auto &patch_type_color_ptr : patch_type)
      Utilities::CUDA::free(patch_type_color_ptr);
    patch_type.clear();

    for (auto &patch_type_color_ptr : patch_type_smooth)
      Utilities::CUDA::free(patch_type_color_ptr);
    patch_type_smooth.clear();

    for (unsigned int d = 0; d < dim; ++d)
      {
        Utilities::CUDA::free(smooth_mass_1d[d]);
        Utilities::CUDA::free(smooth_stiff_1d[d]);
        Utilities::CUDA::free(smooth_mixmass_1d[d]);
        Utilities::CUDA::free(smooth_mixder_1d[d]);

        Utilities::CUDA::free(eigvals[d]);
        Utilities::CUDA::free(eigvecs[d]);
      }

    for (unsigned int d = 0; d < dim; ++d)
      {
        Utilities::CUDA::free(rt_mass_1d[d]);
        Utilities::CUDA::free(rt_laplace_1d[d]);
        Utilities::CUDA::free(mix_mass_1d[d]);
        Utilities::CUDA::free(mix_der_1d[d]);
      }

    Utilities::CUDA::free(inverse_schur);
    Utilities::CUDA::free(vertex_patch_matrices);

    Utilities::CUDA::free(eigenvalues[0]);
    Utilities::CUDA::free(eigenvalues[1]);
    Utilities::CUDA::free(eigenvalues[2]);

    ordering_to_type.clear();
    patch_id_host.clear();
    patch_type_host.clear();
    first_dof_host.clear();
  }

  template <int dim, int fe_degree, typename Number>
  std::size_t
  LevelVertexPatch<dim, fe_degree, Number>::memory_consumption() const
  {
    const unsigned int n_dofs_1d = 2 * fe_degree + 1;

    std::size_t result = 0;

    // For each color, add first_dof, patch_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_patches_laplace[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * (1 << level) * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number>
  std::vector<std::vector<
    typename LevelVertexPatch<dim, fe_degree, Number>::CellIterator>>
  LevelVertexPatch<dim, fe_degree, Number>::gather_vertex_patches(
    const DoFHandler<dim> &dof_handler,
    const unsigned int     level) const
  {
    // LAMBDA checks if a vertex is at the physical boundary
    auto &&is_boundary_vertex = [](const CellIterator &cell,
                                   const unsigned int  vertex_id) {
      return std::any_of(
        std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        [&cell](const auto &face_no) { return cell->at_boundary(face_no); });
    };

    const auto locally_owned_range_mg =
      filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                       IteratorFilters::LocallyOwnedLevelCell());
    /**
     * A mapping @p global_to_local_map between the global vertex and
     * the pair containing the number of locally owned cells and the
     * number of all cells (including ghosts) is constructed
     */
    std::map<unsigned int, std::pair<unsigned int, unsigned int>>
      global_to_local_map;
    for (const auto &cell : locally_owned_range_mg)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if (!is_boundary_vertex(cell, v))
            {
              const unsigned int global_index = cell->vertex_index(v);
              const auto element = global_to_local_map.find(global_index);
              if (element != global_to_local_map.cend())
                {
                  ++(element->second.first);
                  ++(element->second.second);
                }
              else
                {
                  const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
                  const auto status       = global_to_local_map.insert(
                    std::make_pair(global_index, n_cells_pair));
                  (void)status;
                  Assert(status.second,
                         ExcMessage("failed to insert key-value-pair"))
                }
            }
      }

    /**
     * Enumerate the patches contained in @p global_to_local_map by
     * replacing the former number of locally owned cells in terms of a
     * consecutive numbering. The local numbering is required for
     * gathering the level cell iterators into a collection @
     * cell_collections according to the global vertex index.
     */
    unsigned int local_index = 0;
    for (auto &key_value : global_to_local_map)
      {
        key_value.second.first = local_index++;
      }
    const unsigned n_subdomains = global_to_local_map.size();
    AssertDimension(n_subdomains, local_index);
    std::vector<std::vector<CellIterator>> cell_collections;
    cell_collections.resize(n_subdomains);
    for (auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned int global_index = cell->vertex_index(v);
          const auto         element = global_to_local_map.find(global_index);
          if (element != global_to_local_map.cend())
            {
              const unsigned int local_index = element->second.first;
              const unsigned int patch_size  = element->second.second;
              auto              &collection  = cell_collections[local_index];
              if (collection.empty())
                collection.resize(patch_size);
              if (patch_size == regular_vpatch_size) // regular patch
                collection[regular_vpatch_size - 1 - v] = cell;
              else // irregular patch
                AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
            }
        }
    return cell_collections;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::get_patch_data(
    const PatchIterator &patch_v,
    const PatchIterator &patch_p,
    const unsigned int   patch_id)
  {
    const unsigned int n_cell_rt =
      dof_handler_velocity->get_fe().n_dofs_per_cell();
    const unsigned int n_cell_dg =
      dof_handler_pressure->get_fe().n_dofs_per_cell();

    std::vector<unsigned int> local_dof_indices(n_cell_rt + n_cell_dg);

    std::set<unsigned int> dofs_set;
    std::set<unsigned int> dofs_set_p;

    unsigned int it = 0;
    // patch_dofs
    for (unsigned int cell = 0; cell < regular_vpatch_size; ++cell)
      {
        auto cell_ptr_v = (*patch_v)[cell];
        cell_ptr_v->get_mg_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < n_cell_rt; ++i)
          if (dofs_set.find(local_dof_indices[i]) == dofs_set.end())
            {
              patch_dofs_host[patch_id * n_patch_dofs + it] =
                local_dof_indices[i];
              dofs_set.insert(local_dof_indices[i]);

              it++;
            }

        auto cell_ptr_p = (*patch_p)[cell];
        cell_ptr_p->get_mg_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < n_cell_dg; ++i)
          patch_dofs_host[patch_id * n_patch_dofs + n_patch_dofs_rt +
                          cell * n_cell_dg + i] =
            dof_handler_velocity->n_dofs(level) + local_dof_indices[i];


        // for (auto ind : local_dof_indices)
        //   std::cout << ind << " ";
        // std::cout << std::endl;
      }
    AssertDimension(it, n_patch_dofs_rt);

    // std::cout << patch_id << std::endl;
    // for (unsigned int i = n_patch_dofs_rt; i < n_patch_dofs; ++i)
    //   std::cout << patch_dofs_host[patch_id * n_patch_dofs + i] << " ";
    // std::cout << std::endl;

    // patch_type. TODO: Fix: only works on [0,1]^d
    // TODO: level == 1, one patch only.
    const double h            = 1. / Util::pow(2, level);
    auto         first_center = (*patch_v)[0]->center();

    if (level == 1)
      for (unsigned int d = 0; d < dim; ++d)
        patch_type_host[patch_id * dim + d] = 2;
    else
      for (unsigned int d = 0; d < dim; ++d)
        {
          auto pos = std::floor(first_center[d] / h + 1 / 3);
          patch_type_host[patch_id * dim + d] =
            (pos > 0) + (pos == (Util::pow(2, level) - 2));
        }


    // patch_id
    // std::sort(numbering.begin(),
    //           numbering.end(),
    //           [&](unsigned lhs, unsigned rhs) {
    //             return first_dof_host[patch_id * 1 + lhs] <
    //                    first_dof_host[patch_id * 1 + rhs];
    //           });

    // auto encode = [&](unsigned int sum, int val) { return sum * 10 + val; };
    // unsigned int label =
    //   std::accumulate(numbering.begin(), numbering.end(), 0, encode);

    // const auto element = ordering_to_type.find(label);
    // if (element != ordering_to_type.end()) // Fouond
    //   {
    //     patch_id_host[patch_id] = element->second;
    //   }
    // else // Not found
    //   {
    //     ordering_to_type.insert({label, ordering_types++});
    //     patch_id_host[patch_id] = ordering_to_type[label];
    //   }
  }

  template <int dim, int fe_degree, typename Number>
  std::vector<types::global_dof_index>
  get_face_conflicts(
    const typename LevelVertexPatch<dim, fe_degree, Number>::PatchIterator
      &patch)
  {
    std::vector<types::global_dof_index> conflicts;
    for (auto &cell : *patch)
      {
        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          {
            conflicts.push_back(cell->face(face_no)->index());
          }
      }
    return conflicts;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit(
    const DoFHandler<dim>   &mg_dof_v,
    const DoFHandler<dim>   &mg_dof_p,
    const MGConstrainedDoFs &mg_constrained_dofs,
    const unsigned int       mg_level,
    const AdditionalData    &additional_data)
  {
    dof_handler_velocity = &mg_dof_v;
    dof_handler_pressure = &mg_dof_p;

    reinit(mg_dof_v, mg_constrained_dofs, mg_level, additional_data);
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit(
    const DoFHandler<dim>   &mg_dof,
    const MGConstrainedDoFs &mg_constrained_dofs,
    const unsigned int       mg_level,
    const AdditionalData    &additional_data)
  {
    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    this->relaxation         = additional_data.relaxation;
    this->use_coloring       = additional_data.use_coloring;
    this->granularity_scheme = additional_data.granularity_scheme;

    dof_handler = &mg_dof;
    level       = mg_level;

    switch (granularity_scheme)
      {
        case GranularityScheme::none:
          patch_per_block = 1;
          break;
        case GranularityScheme::user_define:
          patch_per_block = additional_data.patch_per_block;
          break;
        case GranularityScheme::multiple:
          patch_per_block = granularity_shmem<dim, fe_degree>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid granularity scheme."));
          break;
      }

    // create patches
    std::vector<std::vector<CellIterator>> cell_collections_velocity;
    cell_collections_velocity =
      std::move(gather_vertex_patches(*dof_handler_velocity, level));

    std::vector<std::vector<CellIterator>> cell_collections_pressure;
    cell_collections_pressure =
      std::move(gather_vertex_patches(*dof_handler_pressure, level));

    graph_ptr_raw_velocity.clear();
    graph_ptr_raw_velocity.resize(1);
    for (auto patch = cell_collections_velocity.begin();
         patch != cell_collections_velocity.end();
         ++patch)
      graph_ptr_raw_velocity[0].push_back(patch);

    graph_ptr_raw_pressure.clear();
    graph_ptr_raw_pressure.resize(1);
    for (auto patch = cell_collections_pressure.begin();
         patch != cell_collections_pressure.end();
         ++patch)
      graph_ptr_raw_pressure[0].push_back(patch);

    // coloring
    graph_ptr_colored_velocity.clear();
    graph_ptr_colored_pressure.clear();
    if (1)
      {
        graph_ptr_colored_velocity.resize(regular_vpatch_size);
        for (auto patch = cell_collections_velocity.begin();
             patch != cell_collections_velocity.end();
             ++patch)
          {
            auto first_cell = (*patch)[0];

            graph_ptr_colored_velocity[first_cell->parent()
                                         ->child_iterator_to_index(first_cell)]
              .push_back(patch);
          }


        graph_ptr_colored_pressure.resize(regular_vpatch_size);
        for (auto patch = cell_collections_pressure.begin();
             patch != cell_collections_pressure.end();
             ++patch)
          {
            auto first_cell = (*patch)[0];

            graph_ptr_colored_pressure[first_cell->parent()
                                         ->child_iterator_to_index(first_cell)]
              .push_back(patch);
          }
      }
    else
      {
        const auto fun = [&](const PatchIterator &filter) {
          return get_face_conflicts<dim, fe_degree, Number>(filter);
        };

        graph_ptr_colored_velocity = std::move(
          GraphColoring::make_graph_coloring(cell_collections_velocity.cbegin(),
                                             cell_collections_velocity.cend(),
                                             fun));

        graph_ptr_colored_pressure = std::move(
          GraphColoring::make_graph_coloring(cell_collections_pressure.cbegin(),
                                             cell_collections_pressure.cend(),
                                             fun));
      }

    if (use_coloring)
      n_colors = graph_ptr_colored_velocity.size();
    else
      n_colors = 1;

    setup_color_arrays(n_colors);

    for (unsigned int i = 0; i < graph_ptr_colored_velocity.size(); ++i)
      {
        auto n_patches      = graph_ptr_colored_velocity[i].size();
        n_patches_smooth[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * 1);

        patch_dofs_host.resize(n_patches * n_patch_dofs);

        auto patch_v   = graph_ptr_colored_velocity[i].begin(),
             end_patch = graph_ptr_colored_velocity[i].end();
        auto patch_p   = graph_ptr_colored_pressure[i].begin();

        for (unsigned int p_id = 0; patch_v != end_patch;
             ++patch_v, ++patch_p, ++p_id)
          get_patch_data(*patch_v, *patch_p, p_id);

        alloc_arrays(&first_dof_smooth[i], n_patches * 1);
        alloc_arrays(&patch_type_smooth[i], n_patches * dim);
        alloc_arrays(&patch_dof_smooth[i], n_patches * n_patch_dofs);

        cudaError_t error_code =
          cudaMemcpy(first_dof_smooth[i],
                     first_dof_host.data(),
                     1 * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_type_smooth[i],
                                patch_type_host.data(),
                                dim * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_dof_smooth[i],
                                patch_dofs_host.data(),
                                n_patch_dofs * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    std::vector<std::vector<PatchIterator>> tmp_ptr;
    tmp_ptr =
      use_coloring ? graph_ptr_colored_velocity : graph_ptr_raw_velocity;

    std::vector<std::vector<PatchIterator>> tmp_ptr_p;
    tmp_ptr_p =
      use_coloring ? graph_ptr_colored_pressure : graph_ptr_raw_pressure;

    ordering_to_type.clear();
    ordering_types = 0;
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto n_patches       = tmp_ptr[i].size();
        n_patches_laplace[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * 1);

        patch_dofs_host.resize(n_patches * n_patch_dofs);

        auto patch = tmp_ptr[i].begin(), end_patch = tmp_ptr[i].end();
        auto patch_p = tmp_ptr_p[i].begin();
        for (unsigned int p_id = 0; patch != end_patch;
             ++patch, ++patch_p, ++p_id)
          get_patch_data(*patch, *patch_p, p_id);

        // alloc_and_copy_arrays(i);
        alloc_arrays(&first_dof_laplace[i], n_patches * 1);
        alloc_arrays(&patch_id[i], n_patches);
        alloc_arrays(&patch_type[i], n_patches * dim);
        alloc_arrays(&patch_dof_laplace[i], n_patches * n_patch_dofs);

        cudaError_t error_code = cudaMemcpy(patch_id[i],
                                            patch_id_host.data(),
                                            n_patches * sizeof(unsigned int),
                                            cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(first_dof_laplace[i],
                                first_dof_host.data(),
                                1 * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        // for (unsigned int j = 0; j < patch_type_host.size(); ++j)
        //   {
        //     std::cout << patch_type_host[j] << " ";
        //     if ((j + 1) % dim == 0)
        //       std::cout << std::endl;
        //   }

        error_code = cudaMemcpy(patch_type[i],
                                patch_type_host.data(),
                                dim * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_dof_laplace[i],
                                patch_dofs_host.data(),
                                n_patch_dofs * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    setup_configuration(n_colors);

    // Mapping
    DoFMapping<dim, fe_degree> dm;

    auto h_interior_host_rt = dm.get_h_to_l_rt_interior();
    auto h_interior_host_dg = dm.get_h_to_l_dg_normal();

    auto htol_rt_host = dm.get_h_to_l_rt();
    auto ltoh_rt_host = dm.get_l_to_h_rt();

    auto htol_rt_interior_host = dm.get_h_to_l_rt_interior();

    auto htol_dgn_host = dm.get_h_to_l_dg_normal();
    auto htol_dgt_host = dm.get_h_to_l_dg_tangent();
    auto htol_dgz_host = dm.get_h_to_l_dg_z();

    auto ltoh_dgn_host = dm.get_l_to_h_dg_normal();
    auto ltoh_dgt_host = dm.get_l_to_h_dg_tangent();
    auto ltoh_dgz_host = dm.get_l_to_h_dg_z();


    std::sort(h_interior_host_rt.begin(), h_interior_host_rt.end());
    std::sort(h_interior_host_dg.begin(), h_interior_host_dg.end());

    for (auto &i : h_interior_host_dg)
      i += n_patch_dofs_rt;

    h_interior_host_rt.insert(h_interior_host_rt.end(),
                              h_interior_host_dg.begin(),
                              h_interior_host_dg.end());

    // for (auto i : h_interior_host_rt)
    //   std::cout << i << " ";

    auto copy_mappings = [](auto &device, const auto &host) {
      cudaError_t cuda_error =
        cudaMemcpyToSymbol(device,
                           host.data(),
                           host.size() * sizeof(unsigned int),
                           0,
                           cudaMemcpyHostToDevice);
      AssertCuda(cuda_error);
    };

    copy_mappings(h_interior, h_interior_host_rt);

    copy_mappings(htol_rt, htol_rt_host);
    // copy_mappings(ltoh_rt, ltoh_rt_host);

    copy_mappings(htol_rt_interior, htol_rt_interior_host);

    copy_mappings(htol_dgn, htol_dgn_host);
    // copy_mappings(htol_dgt, htol_dgt_host);
    // copy_mappings(htol_dgz, htol_dgz_host);

    copy_mappings(ltoh_dgn, ltoh_dgn_host);
    copy_mappings(ltoh_dgt, ltoh_dgt_host);
    copy_mappings(ltoh_dgz, ltoh_dgz_host);


    auto copy_to_device = [](auto &device, const auto &host) {
      LinearAlgebra::ReadWriteVector<unsigned int> rw_vector(host.size());
      device.reinit(host.size());
      for (unsigned int i = 0; i < host.size(); ++i)
        rw_vector[i] = host[i];
      device.import(rw_vector, VectorOperation::insert);
    };

    std::vector<unsigned int> dirichlet_index_vector;
    if (mg_constrained_dofs.have_boundary_indices())
      {
        mg_constrained_dofs.get_boundary_indices(level).fill_index_vector(
          dirichlet_index_vector);
        copy_to_device(dirichlet_indices, dirichlet_index_vector);
      }

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 3;
    constexpr unsigned int n_dofs_2d =
      (2 * fe_degree + 3) * (2 * fe_degree + 3);

    constexpr unsigned int n_patch_dofs =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 1) +
      Util::pow(2 * fe_degree + 2, dim);

    constexpr unsigned int n_patch_dofs_inv =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * (fe_degree + 2) - 3) +
      Util::pow(2 * fe_degree + 2, dim);

    alloc_arrays(&vertex_patch_matrices,
                 Util::pow(n_patch_dofs, 2) * Util::pow(3, dim));

    alloc_arrays(&inverse_schur,
                 Util::pow(n_patch_dofs_dg, 2) * Util::pow(3, dim));

    alloc_arrays(&eigenvalues[0],
                 Util::pow(n_patch_dofs_inv, 2) * Util::pow(3, dim));
    alloc_arrays(&eigenvalues[1],
                 Util::pow(n_patch_dofs_inv, 2) * Util::pow(3, dim));
    alloc_arrays(&eigenvalues[2],
                 Util::pow(n_patch_dofs_inv, 2) * Util::pow(3, dim));


    for (unsigned int d = 0; d < dim; ++d)
      {
        alloc_arrays(&smooth_mass_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&smooth_stiff_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&smooth_mixmass_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&smooth_mixder_1d[d], n_dofs_2d * 3 * dim);

        alloc_arrays(&rt_mass_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&rt_laplace_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&mix_mass_1d[d], n_dofs_2d * 3 * dim);
        alloc_arrays(&mix_der_1d[d], n_dofs_2d * 3 * dim);

        alloc_arrays(&eigvals[d], n_dofs_1d * dim * Util::pow(3, dim));
        alloc_arrays(&eigvecs[d], n_dofs_2d * dim * Util::pow(3, dim));
      }

    reinit_tensor_product_laplace();
    // Timer time;
    reinit_tensor_product_smoother();
    // std::cout << time.wall_time() << std::endl;
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_laplace_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_dofs_per_dim  = (1 << level) * fe_degree + 1;
    data_copy.n_patches       = n_patches_laplace[color];
    data_copy.patch_per_block = patch_per_block;
    data_copy.first_dof       = first_dof_laplace[color];
    data_copy.patch_id        = patch_id[color];
    data_copy.patch_type      = patch_type[color];
    data_copy.rt_mass_1d      = rt_mass_1d;
    data_copy.rt_laplace_1d   = rt_laplace_1d;
    data_copy.mix_mass_1d     = mix_mass_1d;
    data_copy.mix_der_1d      = mix_der_1d;

    data_copy.patch_dof_laplace     = patch_dof_laplace[color];
    data_copy.vertex_patch_matrices = vertex_patch_matrices;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<typename LevelVertexPatch<dim, fe_degree, Number>::Data, 4>
  LevelVertexPatch<dim, fe_degree, Number>::get_smooth_data(
    unsigned int color) const
  {
    std::array<Data, 4> data_copy;

    for (unsigned int i = 0; i < 4; ++i)
      {
        data_copy[i].n_patches         = n_patches_smooth[color];
        data_copy[i].patch_per_block   = patch_per_block;
        data_copy[i].relaxation        = relaxation;
        data_copy[i].first_dof         = first_dof_smooth[color];
        data_copy[i].patch_type        = patch_type_smooth[color];
        data_copy[i].eigenvalues       = eigenvalues[i];
        data_copy[i].eigenvectors      = eigenvectors[i];
        data_copy[i].smooth_mass_1d    = smooth_mass_1d;
        data_copy[i].smooth_stiff_1d   = smooth_stiff_1d;
        data_copy[i].smooth_mixmass_1d = smooth_mixmass_1d;
        data_copy[i].smooth_mixder_1d  = smooth_mixder_1d;

        data_copy[i].eigvals       = eigvals;
        data_copy[i].eigvecs       = eigvecs;
        data_copy[i].inverse_schur = inverse_schur;

        data_copy[i].patch_dof_smooth = patch_dof_smooth[color];
      }

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::patch_loop(const Operator   &op,
                                                       const VectorType &src,
                                                       VectorType &dst) const
  {
    op.setup_kernel(patch_per_block);

    for (unsigned int i = 0; i < graph_ptr_colored_velocity.size(); ++i)
      if (n_patches_smooth[i] > 0)
        {
          op.loop_kernel(src,
                         dst,
                         get_smooth_data(i),
                         grid_dim_smooth[i],
                         block_dim_smooth[i]);

          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::cell_loop(const Operator   &op,
                                                      const VectorType &src,
                                                      VectorType &dst) const
  {
    op.setup_kernel(patch_per_block);

    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches_laplace[i] > 0)
        {
          op.loop_kernel(src,
                         dst,
                         get_laplace_data(i),
                         grid_dim_lapalce[i],
                         block_dim_laplace[i]);

          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_smoother()
    const
  {
    auto RT_mass    = assemble_RTmass_tensor();
    auto RT_laplace = assemble_RTlaplace_tensor();
    auto Mix_mass   = assemble_Mixmass_tensor();
    auto Mix_der    = assemble_Mixder_tensor();

    auto copy_to_device = [](auto tensor, auto dst, unsigned int n) {
      for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int n_elements = Util::pow(2 * fe_degree + 3, 2);

          auto mat = new Number[n_elements * n];
          for (unsigned int i = 0; i < n; ++i)
            std::transform(tensor[d][i].begin(),
                           tensor[d][i].end(),
                           &mat[n_elements * i],
                           [](auto m) -> Number { return m; });

          cudaError_t error_code = cudaMemcpy(dst[d],
                                              mat,
                                              n * n_elements * sizeof(Number),
                                              cudaMemcpyHostToDevice);
          AssertCuda(error_code);

          delete[] mat;
        }
    };

    auto interior =
      [](auto matrix, unsigned int s, unsigned int e, unsigned int o) {
        std::array<std::vector<Table<2, Number>>, dim> dst;

        for (unsigned int d = 0; d < dim; ++d)
          {
            dst[d].resize(e - s);
            if (d == 0)
              for (unsigned int m = s; m < e; ++m)
                {
                  dst[d][m - s].reinit(matrix[d][m].n_rows() - 2,
                                       matrix[d][m].n_cols() - o);

                  for (unsigned int i = 0; i < matrix[d][m].n_rows() - 2; ++i)
                    for (unsigned int j = 0; j < matrix[d][m].n_cols() - o; ++j)
                      dst[d][m - s](i, j) = matrix[d][m](i + 1, j + o / 2);
                }
            else
              for (unsigned int m = s; m < e; ++m)
                {
                  dst[d][m - s].reinit(matrix[d][m].n_rows(),
                                       matrix[d][m].n_cols());

                  for (unsigned int i = 0; i < matrix[d][m].n_rows(); ++i)
                    for (unsigned int j = 0; j < matrix[d][m].n_cols(); ++j)
                      dst[d][m - s](i, j) = matrix[d][m](i, j);
                }
          }
        return dst;
      };

    auto rt_mass_int    = interior(RT_mass, 2, 3, 2);
    auto rt_laplace_int = interior(RT_laplace, 3, 6, 2);
    auto mix_mass_int   = interior(Mix_mass, 2, 3, 0);
    auto mix_der_int    = interior(Mix_der, 2, 3, 0);

    copy_to_device(rt_mass_int, smooth_mass_1d, 1);
    copy_to_device(rt_laplace_int, smooth_stiff_1d, 3);
    copy_to_device(mix_mass_int, smooth_mixmass_1d, 1);
    copy_to_device(mix_der_int, smooth_mixder_1d, 1);


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

    // print_matrices(mix_der_int[0]);
    // print_matrices(mix_mass_int[1]);

    auto copy_vals = [](auto tensor, auto dst, auto shift) {
      constexpr unsigned int n_dofs_1d = Util::pow(2 * fe_degree + 3, 1);

      auto mat = new Number[n_dofs_1d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_1d * i],
                       [](auto m) -> Number { return m; });

      cudaError_t error_code = cudaMemcpy(dst + shift * n_dofs_1d * dim,
                                          mat,
                                          dim * n_dofs_1d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    auto copy_vecs = [](auto tensor, auto dst, auto shift) {
      constexpr unsigned int n_dofs_2d = Util::pow(2 * fe_degree + 3, 2);

      auto mat = new Number[n_dofs_2d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_2d * i],
                       [](auto m) -> Number { return m; });

      cudaError_t error_code = cudaMemcpy(dst + shift * n_dofs_2d * dim,
                                          mat,
                                          dim * n_dofs_2d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    auto fast_diag = [&](auto indices, auto dir) {
      std::array<Table<2, Number>, dim> patch_mass_inv;
      std::array<Table<2, Number>, dim> patch_laplace_inv;

      for (unsigned int d = 0; d < dim; ++d)
        {
          patch_mass_inv[d]    = rt_mass_int[d][0];
          patch_laplace_inv[d] = d == 0 ? rt_laplace_int[d][indices[0]] :
                                 d == 1 ? rt_laplace_int[d][indices[1]] :
                                          rt_laplace_int[d][indices[2]];
        }

      TensorProductData<dim, fe_degree, Number> tensor_product;
      tensor_product.reinit(patch_mass_inv, patch_laplace_inv);

      std::array<AlignedVector<Number>, dim> eigenvalue_tensor;
      std::array<Table<2, Number>, dim>      eigenvector_tensor;
      tensor_product.get_eigenvalues(eigenvalue_tensor);
      tensor_product.get_eigenvectors(eigenvector_tensor);

      auto shift = dir == 0 ? indices[0] + indices[1] * 3 + indices[2] * 9 :
                   dir == 1 ? indices[1] + indices[0] * 3 + indices[2] * 9 :
                              indices[1] + indices[2] * 3 + indices[0] * 9;

      // if (indices[0] * indices[1] == 4)
      //   {
      //     std::cout << "TESTING EIGS " << dir << std::endl;
      //     for (auto eigs : eigenvalue_tensor)
      //       {
      //         for (auto e : eigs)
      //           std::cout << e << " ";
      //         std::cout << std::endl;
      //       }
      //     print_matrices(eigenvector_tensor);
      //   }

      copy_vals(eigenvalue_tensor, eigvals[dir], shift);
      copy_vecs(eigenvector_tensor, eigvecs[dir], shift);
    };

    constexpr unsigned dim_z = dim == 2 ? 1 : 3;

#pragma omp parallel for collapse(3) num_threads(dim_z * 3 * 3) schedule(static)
    for (unsigned int z = 0; z < dim_z; ++z)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          {
            // Exact
            {
              std::array<FullMatrix<double>, dim> A;
              if constexpr (dim == 2)
                {
                  {
                    FullMatrix<double> t0 =
                      Tensors::kronecker_product_(RT_mass[1][2],
                                                  RT_laplace[0][3 + k]);
                    FullMatrix<double> t1 =
                      Tensors::kronecker_product_(RT_laplace[1][3 + j],
                                                  RT_mass[0][2]);
                    t0.add(1., t1);
                    A[0].copy_from(t0);
                  }

                  {
                    FullMatrix<double> t0 =
                      Tensors::kronecker_product_(RT_mass[1][2],
                                                  RT_laplace[0][3 + j]);
                    FullMatrix<double> t1 =
                      Tensors::kronecker_product_(RT_laplace[1][3 + k],
                                                  RT_mass[0][2]);
                    t0.add(1., t1);
                    A[1].copy_from(t0);
                  }
                }
              else if constexpr (dim == 3)
                {
                  {
                    FullMatrix<double> t0 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_mass[1][2],
                                                  RT_laplace[0][3 + k]);
                    FullMatrix<double> t1 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_laplace[1][3 + j],
                                                  RT_mass[0][2]);
                    FullMatrix<double> t2 =
                      Tensors::kronecker_product_(RT_laplace[2][3 + z],
                                                  RT_mass[1][2],
                                                  RT_mass[0][2]);
                    t0.add(1., t1);
                    t2.add(1., t0);
                    A[0].copy_from(t2);
                  }

                  {
                    FullMatrix<double> t0 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_mass[1][2],
                                                  RT_laplace[0][3 + j]);
                    FullMatrix<double> t1 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_laplace[1][3 + k],
                                                  RT_mass[0][2]);
                    FullMatrix<double> t2 =
                      Tensors::kronecker_product_(RT_laplace[2][3 + z],
                                                  RT_mass[1][2],
                                                  RT_mass[0][2]);
                    t0.add(1., t1);
                    t2.add(1., t0);
                    A[1].copy_from(t2);
                  }

                  {
                    FullMatrix<double> t0 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_mass[1][2],
                                                  RT_laplace[0][3 + z]);
                    FullMatrix<double> t1 =
                      Tensors::kronecker_product_(RT_mass[2][2],
                                                  RT_laplace[1][3 + k],
                                                  RT_mass[0][2]);
                    FullMatrix<double> t2 =
                      Tensors::kronecker_product_(RT_laplace[2][3 + j],
                                                  RT_mass[1][2],
                                                  RT_mass[0][2]);
                    t0.add(1., t1);
                    t2.add(1., t0);
                    A[2].copy_from(t2);
                  }
                }

              FullMatrix<double> B =
                dim == 2 ?
                  Tensors::kronecker_product_(Mix_mass[1][2], Mix_der[0][2]) :
                  Tensors::kronecker_product_(Mix_mass[2][2],
                                              Mix_mass[1][2],
                                              Mix_der[0][2]);

              B /= -1;

              FullMatrix<double> Bt;
              Bt.copy_transposed(B);

              const unsigned int n_patch_dofs_inv =
                dim * Util::pow(2 * fe_degree + 2, dim - 1) *
                  (2 * (fe_degree + 2) - 3) +
                Util::pow(2 * fe_degree + 2, dim);

              AssertDimension(A[0].n() * dim + B.n(), n_patch_dofs);

              FullMatrix<double> PatchMatrix(n_patch_dofs, n_patch_dofs);

              for (unsigned int d = 0; d < dim; ++d)
                {
                  PatchMatrix.fill(A[d], d * A[0].m(), d * A[0].n(), 0, 0);
                  PatchMatrix.fill(B, d * A[0].m(), dim * A[0].n(), 0, 0);
                  PatchMatrix.fill(Bt, dim * A[0].m(), d * A[0].n(), 0, 0);
                }

              // PatchMatrix.fill(A[0], 0, 0, 0, 0);
              // PatchMatrix.fill(A[1], A[0].m(), A[0].n(), 0, 0);
              // PatchMatrix.fill(B, 0, dim * A[0].n(), 0, 0);
              // PatchMatrix.fill(B, A[0].m(), dim * A[0].n(), 0, 0);
              // PatchMatrix.fill(Bt, dim * A[0].m(), 0, 0, 0);
              // PatchMatrix.fill(Bt, dim * A[0].m(), A[0].n(), 0, 0);

              DoFMapping<dim, fe_degree> dm;

              auto ind_v_b  = dm.get_l_to_h_rt();
              auto ind_p1   = dm.get_h_to_l_dg_normal();
              auto ind_p1_b = dm.get_l_to_h_dg_normal();
              auto ind_p2_b = dm.get_l_to_h_dg_tangent();
              auto ind_p3_b = dm.get_l_to_h_dg_z();       // indices change 

              FullMatrix<double> AA(PatchMatrix.m(), PatchMatrix.n());   // AA, Bt are submatrices of PatchMatrix

              for (auto i = 0U; i < ind_v_b.size(); ++i)
                for (auto j = 0U; j < ind_v_b.size(); ++j)
                  AA(i, j) = PatchMatrix(ind_v_b[i], ind_v_b[j]);

              for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                for (auto j = 0U; j < ind_p2_b.size(); ++j)
                  Bt(j, i) = B(i, ind_p2_b[j]);

              for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                for (auto j = 0U; j < ind_p1.size(); ++j)
                  PatchMatrix(i + ind_v_b.size() / dim, j + ind_v_b.size()) =
                    Bt(ind_p1[j], i);

              if constexpr (dim == 3)
                {
                  for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                    for (auto j = 0U; j < ind_p3_b.size(); ++j)
                      Bt(j, i) = B(i, ind_p3_b[j]);

                  for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                    for (auto j = 0U; j < ind_p1.size(); ++j)
                      PatchMatrix(i + 2 * ind_v_b.size() / dim,
                                  j + ind_v_b.size()) = Bt(ind_p1[j], i);
                }

              for (auto i = 0U; i < ind_v_b.size(); ++i)
                for (auto j = 0U; j < ind_p1_b.size(); ++j)
                  {
                    AA(i, j + ind_v_b.size()) =
                      PatchMatrix(ind_v_b[i], ind_p1_b[j] + ind_v_b.size());
                    AA(j + ind_v_b.size(), i) = AA(i, j + ind_v_b.size());
                  }

              auto h_interior_host_rt = dm.get_h_to_l_rt_interior();
              auto h_interior_host_dg = dm.get_h_to_l_dg_normal();

              std::sort(h_interior_host_rt.begin(), h_interior_host_rt.end());
              std::sort(h_interior_host_dg.begin(), h_interior_host_dg.end());

              for (auto &i : h_interior_host_dg)
                i += n_patch_dofs_rt;

              h_interior_host_rt.insert(h_interior_host_rt.end(),
                                        h_interior_host_dg.begin(),
                                        h_interior_host_dg.end());

              FullMatrix<double> AA_inv(n_patch_dofs_inv, n_patch_dofs_inv);

              AA_inv.extract_submatrix_from(AA,
                                            h_interior_host_rt,
                                            h_interior_host_rt);

              // if (k == 2 && j == 2)
              //   {
              //     std::ofstream out;
              //     out.open("AA_" + std::to_string(level));
              //     AA.print_formatted(out, 3, true, 0, "0");
              //     out.close();
              //   }

              LAPACKFullMatrix<double> exact_inverse(AA_inv.m(), AA_inv.n()); // AA_inv 是花体A的interior dofs的逆
              exact_inverse = AA_inv;   // 如果只训练单层矩阵；只改这个地方就可以
              // Timer time;
              exact_inverse.compute_inverse_svd_with_kernel(1);
              // std::cout << k + j * 3 + z * 9 << " " << time.wall_time() <<
              // std::endl;
              Vector<double> tmp(AA_inv.m());
              Vector<double> dst(AA_inv.m());
              for (unsigned int col = 0; col < exact_inverse.n(); ++col)
                {
                  tmp[col] = 1;
                  exact_inverse.vmult(dst, tmp);
                  for (unsigned int row = 0; row < exact_inverse.n(); ++row)
                    AA_inv(row, col) = dst[row];
                  tmp[col] = 0;
                }

              // direct
              {
                auto *vals = new Number[AA_inv.m() * AA_inv.n()];

                for (unsigned int r = 0; r < AA_inv.m(); ++r)
                  std::transform(AA_inv.begin(r),
                                 AA_inv.end(r),
                                 &vals[r * AA_inv.n()],
                                 [](auto m) -> Number { return m; });

                cudaError_t error_code =
                  cudaMemcpy(eigenvalues[0] +
                               (k + j * 3 + z * 9) * AA_inv.n_elements(),
                             vals,
                             AA_inv.n_elements() * sizeof(Number),
                             cudaMemcpyHostToDevice);
                AssertCuda(error_code);

                delete[] vals;
              }

              /*
              // Schur direct
              {
                auto h_interior_rt = dm.get_h_to_l_rt_interior();
                auto h_interior_dg = dm.get_h_to_l_dg_normal();

                std::sort(h_interior_rt.begin(), h_interior_rt.end());
                std::sort(h_interior_dg.begin(), h_interior_dg.end());

                for (auto &i : h_interior_dg)
                  i += n_patch_dofs_rt;

                FullMatrix<double> A00(h_interior_rt.size());
                A00.extract_submatrix_from(AA, h_interior_rt, h_interior_rt);

                FullMatrix<double> A01(h_interior_rt.size(),
                                       h_interior_dg.size());
                A01.extract_submatrix_from(AA, h_interior_rt, h_interior_dg);

                FullMatrix<double> A10;
                A10.copy_transposed(A01);

                FullMatrix<double> A00inv(h_interior_rt.size());
                A00inv.invert(A00);

                FullMatrix<double> SchurMatrix(h_interior_dg.size());
                SchurMatrix.triple_product(A00inv, A10, A01);

                LAPACKFullMatrix<double> SchurInv(SchurMatrix.m());
                SchurInv = SchurMatrix;
                SchurInv.compute_inverse_svd_with_kernel(1);

                // if (k == 2 && j == 2)
                //   {
                //     {
                //       std::ofstream out;
                //       out.open("AA_inv_" + std::to_string(level));
                //       AA_inv.print_formatted(out, 3, true, 0, "0");
                //       out.close();
                //     }
                //     {
                //       std::ofstream out;
                //       out.open("A00_" + std::to_string(level));
                //       A00.print_formatted(out, 3, true, 0, "0");
                //       out.close();
                //     }
                //     {
                //       std::ofstream out;
                //       out.open("A00inv_" + std::to_string(level));
                //       A00inv.print_formatted(out, 3, true, 0, "0");
                //       out.close();
                //     }
                //     {
                //       std::ofstream out;
                //       out.open("A01_" + std::to_string(level));
                //       A01.print_formatted(out, 3, true, 0, "0");
                //       out.close();
                //     }
                //     {
                //       std::ofstream out;
                //       out.open("SchurMatrix_" + std::to_string(level));
                //       SchurMatrix.print_formatted(out, 3, true, 0, "0");
                //       out.close();
                //     }
                //   }

                Vector<double> tmp(SchurMatrix.m());
                Vector<double> dst(SchurMatrix.m());
                for (unsigned int col = 0; col < SchurMatrix.n(); ++col)
                  {
                    tmp[col] = 1;
                    SchurInv.vmult(dst, tmp);
                    for (unsigned int row = 0; row < SchurMatrix.n(); ++row)
                      SchurMatrix(row, col) = dst[row];
                    tmp[col] = 0;
                  }

                // if (k == 2 && j == 2)
                //   {
                //     std::ofstream out;
                //     out.open("SchurInv_" + std::to_string(level));
                //     SchurMatrix.print_formatted(out, 3, true, 0, "0");
                //     out.close();
                //   }

                auto *vals  = new Number[AA_inv.m() * AA_inv.n()];
                auto *schur = new Number[SchurMatrix.m() * SchurMatrix.n()];

                for (unsigned int r = 0; r < A00inv.m(); ++r)
                  std::transform(A00inv.begin(r),
                                 A00inv.end(r),
                                 &vals[r * A00inv.n()],
                                 [](auto m) -> Number { return m; });

                for (unsigned int r = 0; r < SchurMatrix.m(); ++r)
                  {
                    std::transform(
                      SchurMatrix.begin(r),
                      SchurMatrix.end(r),
                      &vals[A00inv.n_elements() + r * SchurMatrix.n()],
                      [](auto m) -> Number { return m; });

                    std::transform(SchurMatrix.begin(r),
                                   SchurMatrix.end(r),
                                   &schur[r * SchurMatrix.n()],
                                   [](auto m) -> Number { return m; });
                  }

                for (unsigned int r = 0; r < A01.m(); ++r)
                  std::transform(A01.begin(r),
                                 A01.end(r),
                                 &vals[A00inv.n_elements() +
                                       SchurMatrix.n_elements() + r * A01.n()],
                                 [](auto m) -> Number { return m; });

                for (unsigned int r = 0; r < A10.m(); ++r)
                  std::transform(
                    A10.begin(r),
                    A10.end(r),
                    &vals[A00inv.n_elements() + SchurMatrix.n_elements() +
                          A01.n_elements() + r * A10.n()],
                    [](auto m) -> Number { return m; });

                cudaError_t error_code =
                  cudaMemcpy(eigenvalues[1] +
                               (k + j * 3 + z * 9) * AA_inv.n_elements(),
                             vals,
                             AA_inv.n_elements() * sizeof(Number),
                             cudaMemcpyHostToDevice);
                AssertCuda(error_code);

                error_code =
                  cudaMemcpy(inverse_schur +
                               (k + j * 3 + z * 9) * SchurMatrix.n_elements(),
                             schur,
                             SchurMatrix.n_elements() * sizeof(Number),
                             cudaMemcpyHostToDevice);
                AssertCuda(error_code);

                delete[] vals;
              }

              // Schur Iterative
              {
                auto h_interior_rt = dm.get_h_to_l_rt_interior();
                auto h_interior_dg = dm.get_h_to_l_dg_normal();

                std::sort(h_interior_rt.begin(), h_interior_rt.end());
                std::sort(h_interior_dg.begin(), h_interior_dg.end());

                for (auto &i : h_interior_dg)
                  i += n_patch_dofs_rt;

                FullMatrix<double> A00(h_interior_rt.size());
                A00.extract_submatrix_from(AA, h_interior_rt, h_interior_rt);

                FullMatrix<double> A01(h_interior_rt.size(),
                                       h_interior_dg.size());
                A01.extract_submatrix_from(AA, h_interior_rt, h_interior_dg);

                FullMatrix<double> A10;
                A10.copy_transposed(A01);

                FullMatrix<double> A00inv(h_interior_rt.size());
                A00inv.invert(A00);

                FullMatrix<double> SchurMatrix(h_interior_dg.size());
                SchurMatrix.triple_product(A00inv, A10, A01);

                // LAPACKFullMatrix<Number> SchurInv(SchurMatrix.m());
                // SchurInv = SchurMatrix;
                // SchurInv.compute_inverse_svd_with_kernel(1);

                // Vector<Number> tmp(SchurMatrix.m());
                // Vector<Number> dst(SchurMatrix.m());
                // for (unsigned int col = 0; col < SchurMatrix.n(); ++col)
                //   {
                //     tmp[col] = 1;
                //     SchurInv.vmult(dst, tmp);
                //     for (unsigned int row = 0; row < SchurMatrix.n(); ++row)
                //       SchurMatrix(row, col) = dst[row];
                //     tmp[col] = 0;
                //   }

                auto *vals = new Number[AA_inv.m() * AA_inv.n()];

                for (unsigned int r = 0; r < A00inv.m(); ++r)
                  std::transform(A00inv.begin(r),
                                 A00inv.end(r),
                                 &vals[r * A00inv.n()],
                                 [](auto m) -> Number { return m; });

                for (unsigned int r = 0; r < SchurMatrix.m(); ++r)
                  std::transform(
                    SchurMatrix.begin(r),
                    SchurMatrix.end(r),
                    &vals[A00inv.n_elements() + r * SchurMatrix.n()],
                    [](auto m) -> Number { return m; });

                for (unsigned int r = 0; r < A01.m(); ++r)
                  std::transform(A01.begin(r),
                                 A01.end(r),
                                 &vals[A00inv.n_elements() +
                                       SchurMatrix.n_elements() + r * A01.n()],
                                 [](auto m) -> Number { return m; });

                for (unsigned int r = 0; r < A10.m(); ++r)
                  std::transform(
                    A10.begin(r),
                    A10.end(r),
                    &vals[A00inv.n_elements() + SchurMatrix.n_elements() +
                          A01.n_elements() + r * A10.n()],
                    [](auto m) -> Number { return m; });

                cudaError_t error_code =
                  cudaMemcpy(eigenvalues[2] +
                               (k + j * 3 + z * 9) * AA_inv.n_elements(),
                             vals,
                             AA_inv.n_elements() * sizeof(Number),
                             cudaMemcpyHostToDevice);
                AssertCuda(error_code);

                delete[] vals;
              }
              */
              // Schur FD
              {
                std::vector<unsigned int> indices0{k, j, z};
                fast_diag(indices0, 0);

                std::vector<unsigned int> indices1{j, k, z};
                fast_diag(indices1, 1);

                if constexpr (dim == 3)
                  {
                    std::vector<unsigned int> indices2{z, k, j};
                    fast_diag(indices2, 2);
                  }
              }
            }
          }
  }


  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_laplace()
    const
  {
    auto RT_mass    = assemble_RTmass_tensor();
    auto RT_laplace = assemble_RTlaplace_tensor();
    auto Mix_mass   = assemble_Mixmass_tensor();
    auto Mix_der    = assemble_Mixder_tensor();

    auto copy_to_device = [](auto tensor, auto dst) {
      for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int n_elements = Util::pow(2 * fe_degree + 3, 2);

          auto mat = new Number[n_elements * 3];
          for (unsigned int i = 0; i < 3; ++i)
            std::transform(tensor[d][i].begin(),
                           tensor[d][i].end(),
                           &mat[n_elements * i],
                           [](auto m) -> Number { return m; });

          cudaError_t error_code = cudaMemcpy(dst[d],
                                              mat,
                                              3 * n_elements * sizeof(Number),
                                              cudaMemcpyHostToDevice);
          AssertCuda(error_code);

          delete[] mat;
        }
    };

    copy_to_device(RT_mass, rt_mass_1d);
    copy_to_device(RT_laplace, rt_laplace_1d);
    copy_to_device(Mix_mass, mix_mass_1d);
    copy_to_device(Mix_der, mix_der_1d);

    constexpr unsigned dim_z = dim == 2 ? 1 : 3;

    for (unsigned int z = 0; z < dim_z; ++z)
      for (unsigned int j = 0; j < 3; ++j)
        for (unsigned int k = 0; k < 3; ++k)
          {
            std::array<FullMatrix<double>, dim> A;

            if constexpr (dim == 2)
              {
                {
                  FullMatrix<double> t0 =
                    Tensors::kronecker_product_(RT_mass[1][j],
                                                RT_laplace[0][k]);
                  FullMatrix<double> t1 =
                    Tensors::kronecker_product_(RT_laplace[1][j],
                                                RT_mass[0][k]);
                  t0.add(1., t1);
                  A[0].copy_from(t0);
                }

                {
                  FullMatrix<double> t0 =
                    Tensors::kronecker_product_(RT_mass[1][k],
                                                RT_laplace[0][j]);
                  FullMatrix<double> t1 =
                    Tensors::kronecker_product_(RT_laplace[1][k],
                                                RT_mass[0][j]);
                  t0.add(1., t1);
                  A[1].copy_from(t0);
                }
              }
            else if constexpr (dim == 3)
              {
                {
                  FullMatrix<double> t0 =
                    Tensors::kronecker_product_(RT_mass[2][z],
                                                RT_mass[1][j],
                                                RT_laplace[0][k]);
                  FullMatrix<double> t1 =
                    Tensors::kronecker_product_(RT_mass[2][z],
                                                RT_laplace[1][j],
                                                RT_mass[0][k]);
                  FullMatrix<double> t2 =
                    Tensors::kronecker_product_(RT_laplace[2][z],
                                                RT_mass[1][j],
                                                RT_mass[0][k]);
                  t0.add(1., t1);
                  t2.add(1., t0);
                  A[0].copy_from(t2);
                }

                {
                  FullMatrix<double> t0 =
                    Tensors::kronecker_product_(RT_mass[2][z],
                                                RT_mass[1][k],
                                                RT_laplace[0][j]);
                  FullMatrix<double> t1 =
                    Tensors::kronecker_product_(RT_mass[2][z],
                                                RT_laplace[1][k],
                                                RT_mass[0][j]);
                  FullMatrix<double> t2 =
                    Tensors::kronecker_product_(RT_laplace[2][z],
                                                RT_mass[1][k],
                                                RT_mass[0][j]);
                  t0.add(1., t1);
                  t2.add(1., t0);
                  A[1].copy_from(t2);
                }

                {
                  FullMatrix<double> t0 =
                    Tensors::kronecker_product_(RT_mass[2][j],
                                                RT_mass[1][k],
                                                RT_laplace[0][z]);
                  FullMatrix<double> t1 =
                    Tensors::kronecker_product_(RT_mass[2][j],
                                                RT_laplace[1][k],
                                                RT_mass[0][z]);
                  FullMatrix<double> t2 =
                    Tensors::kronecker_product_(RT_laplace[2][j],
                                                RT_mass[1][k],
                                                RT_mass[0][z]);
                  t0.add(1., t1);
                  t2.add(1., t0);
                  A[2].copy_from(t2);
                }
              }

            // TODO:
            FullMatrix<double> B_x =
              dim == 2 ?
                Tensors::kronecker_product_(Mix_mass[1][j], Mix_der[0][k]) :
                Tensors::kronecker_product_(Mix_mass[2][z],
                                            Mix_mass[1][j],
                                            Mix_der[0][k]);
            FullMatrix<double> B_y =
              dim == 2 ?
                Tensors::kronecker_product_(Mix_mass[1][k], Mix_der[0][j]) :
                Tensors::kronecker_product_(Mix_mass[2][z],
                                            Mix_mass[1][k],
                                            Mix_der[0][j]);
            FullMatrix<double> B_z =
              dim == 2 ?
                Tensors::kronecker_product_(Mix_mass[1][k], Mix_der[0][j]) :
                Tensors::kronecker_product_(Mix_mass[2][j],
                                            Mix_mass[1][k],
                                            Mix_der[0][z]);

            B_x /= -1;
            B_y /= -1;
            B_z /= -1;

            FullMatrix<double> Bt_x;
            FullMatrix<double> Bt_y;
            FullMatrix<double> Bt_z;
            Bt_x.copy_transposed(B_x);
            Bt_y.copy_transposed(B_y);
            Bt_z.copy_transposed(B_z);

            FullMatrix<double> PatchMatrix(A[0].n() * dim + B_x.n(),
                                           A[0].n() * dim + B_x.n());
            PatchMatrix.fill(A[0], 0, 0, 0, 0);
            PatchMatrix.fill(A[1], A[0].m(), A[0].n(), 0, 0);
            PatchMatrix.fill(B_x, 0, dim * A[0].n(), 0, 0);
            PatchMatrix.fill(B_y, A[0].m(), dim * A[0].n(), 0, 0);
            PatchMatrix.fill(Bt_x, dim * A[0].m(), 0, 0, 0);
            PatchMatrix.fill(Bt_y, dim * A[0].m(), A[0].n(), 0, 0);

            if constexpr (dim == 3)
              {
                PatchMatrix.fill(A[2], 2 * A[0].m(), 2 * A[0].n(), 0, 0);
                PatchMatrix.fill(B_z, 2 * A[0].m(), dim * A[0].n(), 0, 0);
                PatchMatrix.fill(Bt_z, dim * A[0].m(), 2 * A[0].n(), 0, 0);
              }

            // std::ofstream out;
            // out.open("patch_mat_L" + std::to_string(level));

            // PatchMatrix.print_formatted(out, 3, true, 0, "0");
            // out.close();

            DoFMapping<dim, fe_degree> dm;

            auto ind_v_b  = dm.get_l_to_h_rt();
            auto ind_p1   = dm.get_h_to_l_dg_normal();
            auto ind_p1_b = dm.get_l_to_h_dg_normal();
            auto ind_p2_b = dm.get_l_to_h_dg_tangent();
            auto ind_p3_b = dm.get_l_to_h_dg_z();

            FullMatrix<double> AA(PatchMatrix.m(), PatchMatrix.n());

            for (auto i = 0U; i < ind_v_b.size(); ++i)
              for (auto j = 0U; j < ind_v_b.size(); ++j)
                AA(i, j) = PatchMatrix(ind_v_b[i], ind_v_b[j]);

            for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
              for (auto j = 0U; j < ind_p2_b.size(); ++j)
                Bt_y(j, i) = B_y(i, ind_p2_b[j]);

            for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
              for (auto j = 0U; j < ind_p1.size(); ++j)
                PatchMatrix(i + ind_v_b.size() / dim, j + ind_v_b.size()) =
                  Bt_y(ind_p1[j], i);

            if constexpr (dim == 3)
              {
                for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                  for (auto j = 0U; j < ind_p3_b.size(); ++j)
                    Bt_z(j, i) = B_z(i, ind_p3_b[j]);

                for (auto i = 0U; i < ind_v_b.size() / dim; ++i)
                  for (auto j = 0U; j < ind_p1.size(); ++j)
                    PatchMatrix(i + 2 * ind_v_b.size() / dim,
                                j + ind_v_b.size()) = Bt_z(ind_p1[j], i);
              }

            for (auto i = 0U; i < ind_v_b.size(); ++i)
              for (auto j = 0U; j < ind_p1_b.size(); ++j)
                {
                  AA(i, j + ind_v_b.size()) =
                    PatchMatrix(ind_v_b[i], ind_p1_b[j] + ind_v_b.size());
                  AA(j + ind_v_b.size(), i) = AA(i, j + ind_v_b.size());
                }

            // std::ofstream out1;
            // out1.open("patch_mat_A_L" + std::to_string(level));

            // AA.print_formatted(out1, 3, true, 0, "0");
            // out1.close();

            auto *vals = new Number[AA.m() * AA.n()];

            for (unsigned int r = 0; r < AA.m(); ++r)
              std::transform(AA.begin(r),
                             AA.end(r),
                             &vals[r * AA.n()],
                             [](auto m) -> Number { return m; });

            cudaError_t error_code =
              cudaMemcpy(vertex_patch_matrices +
                           (k + j * 3 + z * 9) * AA.n_elements(),
                         vals,
                         AA.n_elements() * sizeof(Number),
                         cudaMemcpyHostToDevice);
            AssertCuda(error_code);

            delete[] vals;
          }
  }

  template <int dim, int fe_degree, typename Number>
  std::array<std::array<Table<2, double>, 3>, dim>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_RTmass_tensor() const
  {
    const double h              = Util::pow(2, level);
    const double penalty_factor = h * (fe_degree + 1) * (fe_degree + 2);

    FE_RaviartThomas_new<dim> fe(fe_degree);
    QGauss<1>                 quadrature(fe_degree + 2);

    const unsigned int n_quadrature = quadrature.size();

    std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

    for (unsigned int d = 0; d < dim; ++d)
      {
        n_cell_dofs_1d[d] = d == 0 ? fe_degree + 2 : fe_degree + 1;
        n_patch_dofs_1d[d] =
          d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
      }
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    shape_info.reinit(quadrature, fe);

    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data;
    for (auto d = 0U; d < dim; ++d)
      shape_data[d] = shape_info.get_shape_data(d, 0);

    auto cell_mass = [&](unsigned int pos) {
      std::array<Table<2, double>, dim> mass_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        mass_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

      unsigned int is_first = pos == 0 ? 1 : 0;

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
            {
              double sum_mass = 0;
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

    auto patch_mass = [&](auto left, auto right, auto d) {
      Table<2, double> mass_matrices;

      mass_matrices.reinit(n_patch_dofs_1d[d], n_patch_dofs_1d[d]);

      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            unsigned int shift = d == 0;
            mass_matrices(i, j) += left(i, j);
            mass_matrices(i + n_cell_dofs_1d[d] - shift,
                          j + n_cell_dofs_1d[d] - shift) += right(i, j);
          }
      return mass_matrices;
    };

    auto cell_left  = cell_mass(0);
    auto cell_right = cell_mass(1);

    std::array<std::array<Table<2, double>, 3>, dim> patch_mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass_matrices[d][0] = patch_mass(cell_left[d], cell_right[d], d);
        patch_mass_matrices[d][1] = patch_mass(cell_left[d], cell_right[d], d);
        patch_mass_matrices[d][2] = patch_mass(cell_left[d], cell_left[d], d);
      }

    return patch_mass_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<std::array<Table<2, double>, 6>, dim>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_RTlaplace_tensor() const
  {
    const double h              = Util::pow(2, level);
    const double penalty_factor = h * (fe_degree + 1) * (fe_degree + 2);
    const double scaling_factor = dim == 2 ? 1 : h;

    FE_RaviartThomas_new<dim> fe(fe_degree);
    QGauss<1>                 quadrature(fe_degree + 2);

    const unsigned int n_quadrature = quadrature.size();

    std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

    for (unsigned int d = 0; d < dim; ++d)
      {
        n_cell_dofs_1d[d] = d == 0 ? fe_degree + 2 : fe_degree + 1;
        n_patch_dofs_1d[d] =
          d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
      }
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;
    shape_info.reinit(quadrature, fe);

    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data;
    for (auto d = 0U; d < dim; ++d)
      shape_data[d] = shape_info.get_shape_data(d, 0);

    auto cell_laplace = [&](unsigned int type, unsigned int pos) {
      std::array<Table<2, double>, dim> laplace_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        laplace_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

      unsigned int is_first = pos == 0 ? 1 : 0;

      double boundary_factor_left  = 1.;
      double boundary_factor_right = 1.;

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

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
            {
              double sum_laplace = 0;
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
      std::array<Table<2, double>, dim> mixed_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        mixed_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
            {
              if (d != 0)
                {
                  mixed_matrices[d](j, i) =
                    -0.5 *
                      shape_data[d]
                        .shape_data_on_face[0][i + n_cell_dofs_1d[d]] *
                      shape_data[d].shape_data_on_face[1][j] * h * h +
                    0.5 * shape_data[d].shape_data_on_face[0][i] *
                      shape_data[d]
                        .shape_data_on_face[1][j + n_cell_dofs_1d[d]] *
                      h * h;
                }
            }

      return mixed_matrices;
    };

    auto cell_penalty = [&]() {
      std::array<Table<2, double>, dim> penalty_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        penalty_matrices[d].reinit(n_cell_dofs_1d[d], n_cell_dofs_1d[d]);

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

    auto mixed   = cell_mixed();
    auto penalty = cell_penalty();

    auto patch_laplace = [&](auto left, auto right, auto d) {
      Table<2, double> laplace_matrices;

      laplace_matrices.reinit(n_patch_dofs_1d[d], n_patch_dofs_1d[d]);

      for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
        for (unsigned int j = 0; j < n_cell_dofs_1d[d]; ++j)
          {
            unsigned int shift = d == 0;
            laplace_matrices(i, j) += left(i, j) * scaling_factor;
            laplace_matrices(i + n_cell_dofs_1d[d] - shift,
                             j + n_cell_dofs_1d[d] - shift) +=
              right(i, j) * scaling_factor;

            laplace_matrices(i, j + n_cell_dofs_1d[d] - shift) +=
              mixed[d](i, j) * scaling_factor;
            laplace_matrices(i, j + n_cell_dofs_1d[d] - shift) +=
              penalty[d](i, j) * scaling_factor;

            if (d != 0)
              {
                laplace_matrices(j + n_cell_dofs_1d[d] - shift, i) =
                  laplace_matrices(i, j + n_cell_dofs_1d[d] - shift);
              }
          }
      return laplace_matrices;
    };

    auto cell_left     = cell_laplace(0, 0);
    auto cell_middle_0 = cell_laplace(1, 0);
    auto cell_middle_1 = cell_laplace(1, 1);
    auto cell_right    = cell_laplace(2, 0);

    auto cell_middle = cell_laplace(3, 0);

    std::array<std::array<Table<2, double>, 6>, dim> patch_laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_laplace_matrices[d][0] =
          patch_laplace(cell_left[d], cell_middle_1[d], d);
        patch_laplace_matrices[d][1] =
          patch_laplace(cell_middle_0[d], cell_middle_1[d], d);
        patch_laplace_matrices[d][2] =
          patch_laplace(cell_middle_0[d], cell_right[d], d);

        patch_laplace_matrices[d][3] =
          patch_laplace(cell_left[d], cell_middle[d], d);
        patch_laplace_matrices[d][4] =
          patch_laplace(cell_middle[d], cell_middle[d], d);
        patch_laplace_matrices[d][5] =
          patch_laplace(cell_middle[d], cell_right[d], d);
      }

    if (level == 1)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            patch_laplace_matrices[d][2] =
              patch_laplace(cell_left[d], cell_right[d], d);
            patch_laplace_matrices[d][5] =
              patch_laplace(cell_left[d], cell_right[d], d);
          }
      }

    return patch_laplace_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<std::array<Table<2, double>, 3>, dim>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_Mixmass_tensor() const
  {
    FE_RaviartThomas_new<dim> fe_v(fe_degree);
    FE_DGQLegendre<dim>       fe_p(fe_degree);
    QGauss<1>                 quadrature(fe_degree + 2);

    const unsigned int n_quadrature = quadrature.size();

    std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

    for (unsigned int d = 0; d < dim; ++d)
      {
        n_cell_dofs_1d[d] = d == 0 ? fe_degree + 2 : fe_degree + 1;
        n_patch_dofs_1d[d] =
          d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
      }

    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_v;
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_p;
    shape_info_v.reinit(quadrature, fe_v);
    shape_info_p.reinit(quadrature, fe_p);

    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data_v;
    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data_p;
    for (auto d = 0U; d < dim; ++d)
      {
        shape_data_v[d] = shape_info_v.get_shape_data(d, 0);
        shape_data_p[d] = shape_info_p.get_shape_data(d, 0);
      }

    auto cell_mass = [&](unsigned int pos) {
      std::array<Table<2, double>, dim> mass_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        mass_matrices[d].reinit(n_cell_dofs_1d[d], fe_degree + 1);

      unsigned int is_first = pos == 0 ? 1 : 0;

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < fe_degree + 1; ++j)
            {
              double sum_mass = 0;
              for (unsigned int q = 0; q < n_quadrature; ++q)
                {
                  sum_mass +=
                    shape_data_v[d].shape_values[i * n_quadrature + q] *
                    shape_data_p[d].shape_values[j * n_quadrature + q] *
                    quadrature.weight(q) * is_first;
                }

              mass_matrices[d](i, j) += sum_mass;
            }

      return mass_matrices;
    };

    auto patch_mass = [&](auto left, auto right, auto d) {
      Table<2, double> mass_matrices;

      mass_matrices.reinit(n_patch_dofs_1d[d], 2 * fe_degree + 2);

      if (d != 0)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < fe_degree + 1; ++j)
            {
              unsigned int shift = d == 0;
              mass_matrices(i, j) += left(i, j);
              mass_matrices(i + n_cell_dofs_1d[d] - shift, j + fe_degree + 1) +=
                right(i, j);
            }
      return mass_matrices;
    };

    auto cell_left  = cell_mass(0);
    auto cell_right = cell_mass(1);

    std::array<std::array<Table<2, double>, 3>, dim> patch_mass_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass_matrices[d][0] = patch_mass(cell_left[d], cell_right[d], d);
        patch_mass_matrices[d][1] = patch_mass(cell_left[d], cell_right[d], d);
        patch_mass_matrices[d][2] = patch_mass(cell_left[d], cell_left[d], d);
      }

    return patch_mass_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<std::array<Table<2, double>, 3>, dim>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_Mixder_tensor() const
  {
    FE_RaviartThomas_new<dim> fe_v(fe_degree);
    FE_DGQLegendre<dim>       fe_p(fe_degree);
    QGauss<1>                 quadrature(fe_degree + 2);

    const unsigned int n_quadrature = quadrature.size();

    std::array<unsigned int, dim> n_cell_dofs_1d, n_patch_dofs_1d;

    for (unsigned int d = 0; d < dim; ++d)
      {
        n_cell_dofs_1d[d] = d == 0 ? fe_degree + 2 : fe_degree + 1;
        n_patch_dofs_1d[d] =
          d == 0 ? 2 * n_cell_dofs_1d[d] - 1 : 2 * n_cell_dofs_1d[d];
      }

    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_v;
    internal::MatrixFreeFunctions::ShapeInfo<double> shape_info_p;
    shape_info_v.reinit(quadrature, fe_v);
    shape_info_p.reinit(quadrature, fe_p);

    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data_v;
    std::array<internal::MatrixFreeFunctions::UnivariateShapeData<double>, dim>
      shape_data_p;
    for (auto d = 0U; d < dim; ++d)
      {
        shape_data_v[d] = shape_info_v.get_shape_data(d, 0);
        shape_data_p[d] = shape_info_p.get_shape_data(d, 0);
      }

    auto cell_laplace = [&](unsigned int pos) {
      std::array<Table<2, double>, dim> laplace_matrices;

      for (unsigned int d = 0; d < dim; ++d)
        laplace_matrices[d].reinit(n_cell_dofs_1d[d], fe_degree + 1);

      unsigned int is_first = pos == 0 ? 1 : 0;

      // dir0, mass & laplace
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < fe_degree + 1; ++j)
            {
              double sum_laplace = 0;
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

    auto patch_laplace = [&](auto left, auto right, auto d) {
      Table<2, double> laplace_matrices;

      laplace_matrices.reinit(n_patch_dofs_1d[d], 2 * fe_degree + 2);

      if (d == 0)
        for (unsigned int i = 0; i < n_cell_dofs_1d[d]; ++i)
          for (unsigned int j = 0; j < fe_degree + 1; ++j)
            {
              unsigned int shift = d == 0;
              laplace_matrices(i, j) += left(i, j);
              laplace_matrices(i + n_cell_dofs_1d[d] - shift,
                               j + fe_degree + 1) += right(i, j);
            }
      return laplace_matrices;
    };

    auto cell_left  = cell_laplace(0);
    auto cell_right = cell_laplace(1);

    std::array<std::array<Table<2, double>, 3>, dim> patch_laplace_matrices;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_laplace_matrices[d][0] =
          patch_laplace(cell_left[d], cell_right[d], d);
        patch_laplace_matrices[d][1] =
          patch_laplace(cell_left[d], cell_right[d], d);
        patch_laplace_matrices[d][2] =
          patch_laplace(cell_left[d], cell_left[d], d);
      }

    return patch_laplace_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_color_arrays(
    const unsigned int n_colors)
  {
    this->n_patches_laplace.resize(n_colors);
    this->grid_dim_lapalce.resize(n_colors);
    this->block_dim_laplace.resize(n_colors);
    this->first_dof_laplace.resize(n_colors);
    this->patch_id.resize(n_colors);
    this->patch_type.resize(n_colors);
    this->patch_dof_laplace.resize(n_colors);

    this->n_patches_smooth.resize(graph_ptr_colored_velocity.size());
    this->grid_dim_smooth.resize(graph_ptr_colored_velocity.size());
    this->block_dim_smooth.resize(graph_ptr_colored_velocity.size());
    this->first_dof_smooth.resize(graph_ptr_colored_velocity.size());
    this->patch_type_smooth.resize(graph_ptr_colored_velocity.size());
    this->patch_dof_smooth.resize(graph_ptr_colored_velocity.size());
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_configuration(
    const unsigned int n_colors)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto         n_patches = n_patches_laplace[i];
        const double apply_n_blocks =
          std::ceil(static_cast<double>(n_patches) /
                    static_cast<double>(patch_per_block));

        grid_dim_lapalce[i]  = dim3(apply_n_blocks);
        block_dim_laplace[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
      }

    for (unsigned int i = 0; i < graph_ptr_colored_velocity.size(); ++i)
      {
        auto         n_patches = n_patches_smooth[i];
        const double apply_n_blocks =
          std::ceil(static_cast<double>(n_patches) /
                    static_cast<double>(patch_per_block));

        grid_dim_smooth[i]  = dim3(apply_n_blocks);
        block_dim_smooth[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Number1>
  void
  LevelVertexPatch<dim, fe_degree, Number>::alloc_arrays(Number1 **array_device,
                                                         const unsigned int n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
  }

  template <typename Number>
  __global__ void
  copy_constrained_values_kernel(const Number       *src,
                                 Number             *dst,
                                 const unsigned int *indices,
                                 const unsigned int  len)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
      {
        dst[indices[idx]] = src[indices[idx]];
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::copy_constrained_values(
    const VectorType &src,
    VectorType       &dst) const
  {
    const unsigned int len = dirichlet_indices.size();
    if (len > 0)
      {
        const unsigned int bksize  = 256;
        const unsigned int nblocks = (len - 1) / bksize + 1;
        dim3               bk_dim(bksize);
        dim3               gd_dim(nblocks);

        copy_constrained_values_kernel<<<gd_dim, bk_dim>>>(
          src.get_values(),
          dst.get_values(),
          dirichlet_indices.get_values(),
          len);
        AssertCudaKernel();
      }
  }

  template <typename Number>
  __global__ void
  set_constrained_values_kernel(Number             *dst,
                                const unsigned int *indices,
                                const unsigned int  len)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
      {
        dst[indices[idx]] = 0;
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::set_constrained_values(
    VectorType &dst) const
  {
    const unsigned int len = dirichlet_indices.size();
    if (len > 0)
      {
        const unsigned int bksize  = 256;
        const unsigned int nblocks = (len - 1) / bksize + 1;
        dim3               bk_dim(bksize);
        dim3               gd_dim(nblocks);

        set_constrained_values_kernel<<<gd_dim, bk_dim>>>(
          dst.get_values(), dirichlet_indices.get_values(), len);
        AssertCudaKernel();
      }
  }

} // namespace PSMF

/**
 * \page patch_base.template
 * \include patch_base.template.cuh
 */
