/**
 * @file patch_base.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PATCH_BASE_CUH
#define PATCH_BASE_CUH

#include <deal.II/grid/filtered_iterator.h>

#include "TPSS/kroneckersvd.h"
#include "TPSS/tensors.h"
#include "cuda_vector.cuh"
#include "tensor_product.h"
#include "utilities.cuh"
#include <cuda/std/array>

using namespace dealii;

/**
 * Namespace for the Patch Smoother Matrix-Free
 */
namespace PSMF
{

  /**
   * @brief Laplace Variant: kernel type for Laplace operator.
   */
  enum class LaplaceVariant
  {
    /**
     * Basic implementation.
     */
    Basic,

    /**
     * Basic implementation. Load data cell by cell.
     */
    BasicCell,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore,

    /**
     * Using the Matrix Structure.
     */
    MatrixStruct
  };


  /**
   * @brief Smoother Variant: kernel type for
   * Multiplicative Schwarz Smoother.
   */
  enum class SmootherVariant
  {
    /**
     * Compute the residual globally, i.e.
     * r = b - Ax, where A is the system matrix.
     */
    GLOBAL,

    /**
     * Same as above, but with linear thread indicex instead of tmasking
     * boundary threads for local solver.
     */
    FUSED_L,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore
  };

  enum class LocalSolverVariant
  {
    Direct,
    SchurDirect,
    SchurIter,
    SchurTensorProduct,
    Uzawa
  };


  enum class DoFLayout
  {
    DGQ,
    Q,
    RT
  };

  enum class SolverVariant
  {
    GMRES,
    PCG,
    FMG,
    Linear_FMG,
    Vcycle
  };

  enum class ExpementsSets
  {
    none,
    kernel,
    error_analysis,
    solvers,
    vnum
  };

  /**
   * Granularity Scheme: number of patches per thread-block
   */
  enum class GranularityScheme
  {
    none,
    user_define,
    multiple
  };


  /**
   * @brief Implementation for Discontinuous Galerkin(DG) element
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   */
  template <int dim, int fe_degree, typename Number>
  class LevelVertexPatch : public Subscriptor
  {
  public:
    using CellIterator = typename DoFHandler<dim>::level_cell_iterator;
    using PatchIterator =
      typename std::vector<std::vector<CellIterator>>::const_iterator;

    static constexpr unsigned int regular_vpatch_size = 1 << dim;

    static constexpr unsigned int n_patch_dofs_rt =
      dim * Util::pow(2 * fe_degree + 2, dim - 1) * (2 * fe_degree + 3);

    static constexpr unsigned int n_patch_dofs_dg =
      Util::pow(2 * fe_degree + 2, dim);

    static constexpr unsigned int n_patch_dofs =
      n_patch_dofs_rt + n_patch_dofs_dg;

    /**
     * Standardized data struct to pipe additional data to LevelVertexPatch.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(
        const Number            relaxation         = 1.,
        const bool              use_coloring       = true,
        const unsigned int      patch_per_block    = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none)
        : relaxation(relaxation)
        , use_coloring(use_coloring)
        , patch_per_block(patch_per_block)
        , granularity_scheme(granularity_scheme)
      {}

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * If true, use coloring. Otherwise, use atomic operations.
       * Coloring ensures bitwise reproducibility but is slower on Pascal and
       * newer architectures (need to check).
       */
      bool use_coloring;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      GranularityScheme granularity_scheme;
    };


    /**
     * Structure which is passed to the kernel.
     * It is used to pass all the necessary information from the CPU to the
     * GPU.
     */
    struct Data
    {
      /**
       * Number of dofs per dim.
       */
      unsigned int n_dofs_per_dim;

      /**
       * Number of patches for each color.
       */
      unsigned int n_patches;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * Pointer to the the first degree of freedom in each patch.
       * @note Need Lexicographic ordering degree of freedoms.
       * @note For DG case, the first degree of freedom index of
       *       four cells in a patch is stored consecutively.
       */
      unsigned int *first_dof;

      unsigned int *patch_dof_laplace;
      unsigned int *patch_dof_smooth;

      /**
       * Pointer to patch matrices for Stokes operator.
       */
      Number *vertex_patch_matrices;

      /**
       * Pointer to the patch cell ordering type.
       */
      unsigned int *patch_id;

      /**
       * Pointer to the patch type. left, middle, right
       */
      unsigned int *patch_type;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *l_to_h;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *h_to_l;

      /**
       * Pointer to 1D RT mass matrix for Stokes operator.
       */
      cuda::std::array<Number *, dim> rt_mass_1d;

      /**
       * Pointer to 1D RT stiffness matrix for Stokes operator.
       */
      cuda::std::array<Number *, dim> rt_laplace_1d;

      /**
       * Pointer to 1D mixed mass matrix for Stokes operator.
       */
      cuda::std::array<Number *, dim> mix_mass_1d;

      /**
       * Pointer to 1D mixed derivative matrix for Stokes operator.
       */
      cuda::std::array<Number *, dim> mix_der_1d;

      /**
       * Pointer to 1D RT mass matrix for smoothing operator.
       */
      cuda::std::array<Number *, dim> smooth_mass_1d;

      /**
       * Pointer to 1D RT stiffness matrix for smoothing operator.
       */
      cuda::std::array<Number *, dim> smooth_stiff_1d;

      /**
       * Pointer to 1D mixed mass matrix for smoothing operator.
       */
      cuda::std::array<Number *, dim> smooth_mixmass_1d;

      /**
       * Pointer to 1D mixed derivative matrix for smoothing operator.
       */
      cuda::std::array<Number *, dim> smooth_mixder_1d;

      /**
       * Pointer to 1D eigenvalues for smoothing operator.
       */
      Number *eigenvalues;

      /**
       * Pointer to 1D eigenvectors for smoothing operator.
       */
      Number *eigenvectors;


      /**
       * Pointer to 1D eigenvalues for smoothing operator RT.
       */
      cuda::std::array<Number *, dim> eigvals;

      /**
       * Pointer to 1D eigenvectors for smoothing operator RT.
       */
      cuda::std::array<Number *, dim> eigvecs;

      /**
       * Schur Inv
       */
      Number *inverse_schur;
    };

    /**
     * Default constructor.
     */
    LevelVertexPatch();

    /**
     * Destructor.
     */
    ~LevelVertexPatch();

    /**
     * Return the Data structure associated with @p color for lapalce operator.
     */
    Data
    get_laplace_data(unsigned int color) const;

    /**
     * Return the Data structure associated with @p color for smoothing operator.
     */
    std::array<Data, 4>
    get_smooth_data(unsigned int color) const;

    /**
     * Extracts the information needed to perform loops over cells.
     */
    void
    reinit(const DoFHandler<dim>   &dof_handler,
           const MGConstrainedDoFs &mg_constrained_dofs,
           const unsigned int       mg_level,
           const AdditionalData    &additional_data = AdditionalData());

    void
    reinit(const DoFHandler<dim>   &dof_handler_velocity,
           const DoFHandler<dim>   &dof_handler_pressure,
           const MGConstrainedDoFs &mg_constrained_dofs,
           const unsigned int       mg_level,
           const AdditionalData    &additional_data = AdditionalData());

    /**
     * @brief This method runs the loop over all patches and apply the local operation on
     * each element in parallel.
     *
     * @tparam Operator a operator which is applied on each patch
     * @tparam VectorType
     * @param func
     * @param src
     * @param dst
     */
    template <typename Operator, typename VectorType>
    void
    patch_loop(const Operator   &op,
               const VectorType &src,
               VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and compute tensor product on
     * each element in parallel for global mat-vec operation.
     * We call it 'cell loop' because although patches are used as a unit, only
     * some of the cells are updated, thus avoiding the duplication of face
     * integrals.
     */
    template <typename Operator, typename VectorType>
    void
    cell_loop(const Operator &op, const VectorType &src, VectorType &dst) const;

    /**
     * @brief Initializes the tensor product matrix for local smoothing.
     */
    void
    reinit_tensor_product_smoother() const;

    /**
     * @brief Initializes the tensor product matrix for local laplace.
     */
    void
    reinit_tensor_product_laplace() const;

    /**
     * Copy the values of the constrained entries from src to dst. This is used
     * to impose zero Dirichlet boundary condition.
     */
    template <typename VectorType>
    void
    copy_constrained_values(const VectorType &src, VectorType &dst) const;

    /**
     * Set the values of the constrained entries to zero. This is used
     * to impose zero Dirichlet boundary condition.
     */
    template <typename VectorType>
    void
    set_constrained_values(VectorType &dst) const;

    /**
     * Free all the memory allocated.
     */
    void
    free();

    /**
     * Return an approximation of the memory consumption of this class in
     * bytes.
     */
    std::size_t
    memory_consumption() const;

    /**
     * Helper function. Assemble 1d RT mass matrices.
     */
    std::array<std::array<Table<2, double>, 3>, dim>
    assemble_RTmass_tensor() const;

    /**
     * Helper function. Assemble 1d RT laplace matrices.
     */
    std::array<std::array<Table<2, double>, 6>, dim>
    assemble_RTlaplace_tensor() const;

    /**
     * Helper function. Assemble 1d Mixed mass matrices.
     */
    std::array<std::array<Table<2, double>, 3>, dim>
    assemble_Mixmass_tensor() const;

    /**
     * Helper function. Assemble 1d Mixed derivative matrices.
     */
    std::array<std::array<Table<2, double>, 3>, dim>
    assemble_Mixder_tensor() const;

  private:
    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_color_arrays(const unsigned int n_colors);

    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_configuration(const unsigned int n_colors);

    /**
     * Helper function. Get tensor product data for each patch.
     */
    void
    get_patch_data(const PatchIterator &patch_v,
                   const PatchIterator &patch_p,
                   const unsigned int   patch_id);

    /**
     * Gathering the locally owned and ghost cells attached to a common
     * vertex as the collection of cell iterators (patch). The
     * successive distribution of the collections with ghost cells
     * follows the logic:
     *
     * 1.) if one mpi-proc owns more than half of the cells (locally owned)
     * of the vertex patch the mpi-proc takes ownership
     *
     * 2.) for the remaining ghost patches the mpi-proc with the cell of
     * the lowest CellId (see dealii::Triangulation) takes ownership
     */
    std::vector<std::vector<CellIterator>>
    gather_vertex_patches(const DoFHandler<dim> &dof_handler,
                          const unsigned int     level) const;

    /**
     * Allocate an array to the device.
     */
    template <typename Number1>
    void
    alloc_arrays(Number1 **array_device, const unsigned int n);

    /**
     * Number of global refinments.
     */
    unsigned int level;

    /**
     * Number of colors produced by the coloring algorithm.
     */
    unsigned int n_colors;

    /**
     * Relaxation parameter.
     */
    Number relaxation;

    /**
     * If true, use coloring. Otherwise, use atomic operations.
     * Coloring ensures bitwise reproducibility but is slower on Pascal and
     * newer architectures.
     */
    bool use_coloring;


    GranularityScheme granularity_scheme;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim_lapalce;
    std::vector<dim3> grid_dim_smooth;

    /**
     * Block dimensions associated to the different colors. The block
     * dimensions are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim_laplace;
    std::vector<dim3> block_dim_smooth;

    /**
     * Number of patches per thread block.
     */
    unsigned int patch_per_block;

    /**
     * Auxiliary vector.
     */
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;

    /**
     * Raw graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_raw_velocity;
    std::vector<std::vector<PatchIterator>> graph_ptr_raw_pressure;
    /**
     * Colored graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_colored_velocity;
    std::vector<std::vector<PatchIterator>> graph_ptr_colored_pressure;

    /**
     * Number of patches in each color.
     */
    std::vector<unsigned int> n_patches_laplace;
    std::vector<unsigned int> n_patches_smooth;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    const DoFHandler<dim> *dof_handler_velocity;
    const DoFHandler<dim> *dof_handler_pressure;

    /**
     * Vector of pointer to the the first degree of freedom
     * in each patch of each color.
     * @note Need Lexicographic ordering degree of freedoms.
     * @note For DG case, the first degree of freedom index of
     *       four cells in a patch is stored consecutively.
     */
    std::vector<unsigned int *> first_dof_laplace;
    std::vector<unsigned int *> first_dof_smooth;

    std::vector<unsigned int *> patch_dof_laplace;
    std::vector<unsigned int *> patch_dof_smooth;

    /**
     * Vector of the the first degree of freedom
     * in each patch of a single color.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> first_dof_host;

    std::vector<unsigned int> patch_dofs_host;

    /**
     * Vector of pointer to patch type: left, middle, right.
     */
    std::vector<unsigned int *> patch_type;
    std::vector<unsigned int *> patch_type_smooth;

    /**
     * Vector of patch type: left, middle, right.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_type_host;

    /**
     * Vector of pointer to the local cell ordering type for each patch.
     */
    std::vector<unsigned int *> patch_id;

    /**
     * Vector of patch cell ordering type.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_id_host;

    /**
     * Mapping from cell ordering to type id.
     */
    std::unordered_map<unsigned int, unsigned int> ordering_to_type;

    /**
     * Categories of ordering types.
     */
    unsigned int ordering_types;

    /**
     * Lookup table
     */
    std::unordered_map<unsigned int, std::array<unsigned int, 3>> lookup_table;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *l_to_h;

    std::vector<unsigned int> l_to_h_host;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *h_to_l;

    std::vector<unsigned int> h_to_l_host;

    /**
     * A variable storing the local indices of Dirichlet boundary conditions
     * on cells for all levels (outer index), the cells within the levels
     * (second index), and the indices on the cell (inner index).
     */
    CudaVector<unsigned int> dirichlet_indices;

    /**
     * Pointer to 1D RT mass matrix for Stokes operator.
     */
    cuda::std::array<Number *, dim> rt_mass_1d;

    /**
     * Pointer to 1D RT stiffness matrix for Stokes operator.
     */
    cuda::std::array<Number *, dim> rt_laplace_1d;

    /**
     * Pointer to 1D mixed mass matrix for Stokes operator.
     */
    cuda::std::array<Number *, dim> mix_mass_1d;

    /**
     * Pointer to 1D mixed derivative matrix for Stokes operator.
     */
    cuda::std::array<Number *, dim> mix_der_1d;

    /**
     * Pointer to patch matrices for Stokes operator.
     */
    Number *vertex_patch_matrices;

    /**
     * Pointer to 1D RT mass matrix for smoothing operator.
     */
    cuda::std::array<Number *, dim> smooth_mass_1d;

    /**
     * Pointer to 1D RT stiffness matrix for smoothing operator.
     */
    cuda::std::array<Number *, dim> smooth_stiff_1d;

    /**
     * Pointer to 1D mixed mass matrix for smoothing operator.
     */
    cuda::std::array<Number *, dim> smooth_mixmass_1d;

    /**
     * Pointer to 1D mixed derivative matrix for smoothing operator.
     */
    cuda::std::array<Number *, dim> smooth_mixder_1d;

    /**
     * Pointer to 1D eigenvalues for smoothing operator.
     */
    std::array<Number *, 4> eigenvalues;

    /**
     * Pointer to 1D eigenvectors for smoothing operator.
     */
    std::array<Number *, 4> eigenvectors;

    /**
     * Pointer to 1D eigenvalues for smoothing operator RT.
     */
    cuda::std::array<Number *, dim> eigvals;

    /**
     * Pointer to 1D eigenvectors for smoothing operator RT.
     */
    cuda::std::array<Number *, dim> eigvecs;

    /**
     * Schur Inv
     */
    Number *inverse_schur;
  };

  template <typename Number>
  struct SharedDataBase
  {
    /**
     * Shared memory for local and interior src.
     */
    Number *local_src;

    /**
     * Shared memory for local and interior dst.
     */
    Number *local_dst;

    /**
     * Shared memory for local and interior residual.
     */
    Number *local_residual;

    /**
     * Shared memory for computed 1D mass matrix.
     */
    Number *local_mass;

    /**
     * Shared memory for computed 1D Laplace matrix.
     */
    Number *local_laplace;

    /**
     * Shared memory for computed 1D mixed mass matrix.
     */
    Number *local_mix_mass;

    /**
     * Shared memory for computed 1D mixed derivative matrix.
     */
    Number *local_mix_der;

    /**
     * Shared memory for computed 1D eigenvalues.
     */
    Number *local_eigenvalues;

    /**
     * Shared memory for computed 1D eigenvectors.
     */
    Number *local_eigenvectors;

    /**
     * Shared memory for internal buffer.
     */
    Number *tmp;
  };


  __constant__ unsigned int
    h_interior[Util::MAX_PATCH_DOFS_RT_INT + Util::MAX_PATCH_DOFS_DG];
  __constant__ unsigned int htol_rt_interior[Util::MAX_PATCH_DOFS_RT_INT];

  __constant__ unsigned int htol_rt[Util::MAX_PATCH_DOFS_RT];
  // __constant__ unsigned int ltoh_rt[Util::MAX_PATCH_DOFS_RT];

  __constant__ unsigned int htol_dgn[Util::MAX_PATCH_DOFS_DG];
  // __constant__ unsigned int htol_dgt[Util::MAX_PATCH_DOFS_DG];
  // __constant__ unsigned int htol_dgz[Util::MAX_PATCH_DOFS_DG];

  __constant__ unsigned int ltoh_dgn[Util::MAX_PATCH_DOFS_DG];
  __constant__ unsigned int ltoh_dgt[Util::MAX_PATCH_DOFS_DG];
  __constant__ unsigned int ltoh_dgz[Util::MAX_PATCH_DOFS_DG];

  /**
   * Structure to pass the shared memory into a general user function.
   * Used for Stokes operator.
   */
  template <int dim, typename Number, LaplaceVariant laplace>
  struct SharedDataOp;

  template <int dim, typename Number>
  struct SharedDataOp<dim, Number, LaplaceVariant::MatrixStruct>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;

    __device__
    SharedDataOp(Number      *data,
                 unsigned int n_buff,
                 unsigned int n_dofs_1d,
                 unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;
    }
  };

  template <int dim, typename Number>
  struct SharedDataOp<dim, Number, LaplaceVariant::Basic>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::local_mass;
    using SharedDataBase<Number>::local_laplace;
    using SharedDataBase<Number>::local_mix_mass;
    using SharedDataBase<Number>::local_mix_der;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataOp(Number      *data,
                 unsigned int n_buff,
                 unsigned int n_dofs_1d,
                 unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass    = local_dst + n_buff * local_dim;
      local_laplace = local_mass + n_buff * n_dofs_1d * n_dofs_1d * dim * dim;
      local_mix_mass =
        local_laplace + n_buff * n_dofs_1d * n_dofs_1d * dim * dim;
      local_mix_der =
        local_mix_mass + n_buff * n_dofs_1d * n_dofs_1d * dim * (dim - 1);

      tmp = local_mix_der + n_buff * n_dofs_1d * n_dofs_1d * dim;
    }
  };

  /**
   * Structure to pass the shared memory into a general user function.
   * Used for local smoothing operator.
   */
  template <int dim,
            typename Number,
            SmootherVariant    smoother,
            LocalSolverVariant local_solver>
  struct SharedDataSmoother;

  /**
   * Exact local solver.
   */
  template <int dim, typename Number, SmootherVariant smoother>
  struct SharedDataSmoother<dim, Number, smoother, LocalSolverVariant::Direct>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::local_mass;
    using SharedDataBase<Number>::local_laplace;
    using SharedDataBase<Number>::local_mix_mass;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass     = local_dst + n_buff * local_dim;
      local_laplace  = local_mass + n_buff * n_dofs_1d * n_dofs_1d * dim;
      local_mix_mass = local_laplace + n_buff * n_dofs_1d * n_dofs_1d * dim;

      tmp = local_mix_mass + n_buff * n_dofs_1d * n_dofs_1d * dim;
    }
  };

  template <int dim, typename Number, SmootherVariant smoother>
  struct SharedDataSmoother<dim,
                            Number,
                            smoother,
                            LocalSolverVariant::SchurDirect>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;
      tmp       = local_dst + n_buff * local_dim;
    }
  };

  template <int dim, typename Number, SmootherVariant smoother>
  struct SharedDataSmoother<dim,
                            Number,
                            smoother,
                            LocalSolverVariant::SchurIter>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;
      tmp       = local_dst + n_buff * local_dim;
    }
  };


  template <int dim, typename Number, SmootherVariant smoother>
  struct SharedDataSmoother<dim,
                            Number,
                            smoother,
                            LocalSolverVariant::SchurTensorProduct>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::local_mass;
    using SharedDataBase<Number>::local_laplace;
    using SharedDataBase<Number>::local_mix_mass;
    using SharedDataBase<Number>::local_mix_der;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass    = local_dst + n_buff * local_dim;
      local_laplace = local_mass + n_buff * n_dofs_1d * dim * dim;
      local_mix_mass =
        local_laplace + n_buff * n_dofs_1d * n_dofs_1d * dim * dim;
      local_mix_der =
        local_mix_mass + n_buff * n_dofs_1d * n_dofs_1d * (dim - 1);

      tmp = local_mix_der + n_buff * n_dofs_1d * n_dofs_1d * 1;

      local_vars = tmp + n_buff * 4 * local_dim;
    }

    Number *local_vars;
  };

  template <int dim, typename Number, SmootherVariant smoother>
  struct SharedDataSmoother<dim, Number, smoother, LocalSolverVariant::Uzawa>
    : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::local_mass;
    using SharedDataBase<Number>::local_laplace;
    using SharedDataBase<Number>::local_mix_mass;
    using SharedDataBase<Number>::local_mix_der;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      local_mass    = local_dst + n_buff * local_dim;
      local_laplace = local_mass + n_buff * n_dofs_1d * dim * dim;
      local_mix_mass =
        local_laplace + n_buff * n_dofs_1d * n_dofs_1d * dim * dim;
      local_mix_der =
        local_mix_mass + n_buff * n_dofs_1d * n_dofs_1d * (dim - 1);

      tmp = local_mix_der + n_buff * n_dofs_1d * n_dofs_1d * 1;

      local_vars = tmp + n_buff * 4 * local_dim;
    }

    Number *local_vars;
  };


  /**
   * Bila or KSVD12 local solver.
   */
  template <int dim,
            typename Number,
            SmootherVariant    smoother,
            LocalSolverVariant local_solver>
  struct SharedDataSmoother : SharedDataBase<Number>
  {
    using SharedDataBase<Number>::local_src;
    using SharedDataBase<Number>::local_dst;
    using SharedDataBase<Number>::local_mass;
    using SharedDataBase<Number>::local_laplace;
    using SharedDataBase<Number>::local_mix_mass;
    using SharedDataBase<Number>::tmp;

    __device__
    SharedDataSmoother(Number      *data,
                       unsigned int n_buff,
                       unsigned int n_dofs_1d,
                       unsigned int local_dim)
    {
      local_src = data;
      local_dst = local_src + n_buff * local_dim;

      if constexpr (smoother == SmootherVariant::GLOBAL)
        {
          local_mass    = local_dst + n_buff * local_dim;
          local_laplace = local_mass + n_buff * n_dofs_1d * dim;
          tmp           = local_laplace + n_buff * n_dofs_1d * n_dofs_1d * dim;
        }
      else
        {
          local_mass     = local_dst + n_buff * local_dim;
          local_laplace  = local_mass + n_buff * n_dofs_1d * n_dofs_1d;
          local_mix_mass = local_laplace + n_buff * n_dofs_1d * n_dofs_1d;
          // TODO: should be able to use less shared memory
          tmp = local_mix_mass + n_buff * n_dofs_1d * n_dofs_1d * (dim - 1);
        }
    }
  };


  /**
   * This function determines number of patches per block at compile time.
   */
  template <int dim, int fe_degree>
  __host__ __device__ constexpr unsigned int
  granularity_shmem()
  {
    return dim == 2 ? (fe_degree == 1 ? 8 :
                       fe_degree == 2 ? 4 :
                       fe_degree == 3 ? 4 :
                       fe_degree == 4 ? 2 :
                                        1) :
           dim == 3 ? (fe_degree == 1 ? 16 :
                       fe_degree == 2 ? 2 :
                                        1) :
                      1;
  }

} // namespace PSMF

#include "patch_base.template.cuh"

/**
 * \page patch_base
 * \include patch_base.cuh
 */

#endif // PATCH_BASE_CUH
