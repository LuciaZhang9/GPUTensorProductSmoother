/**
 * @file cuda_mg_transfer.cu
 * Created by Cu Cui on 2022/12/25.
 */

#include <deal.II/lac/la_parallel_vector.templates.h>

#include <deal.II/multigrid/multigrid.templates.h>

#include "cuda_mg_transfer.cuh"
#include "cuda_mg_transfer.template.cuh"

namespace PSMF
{

  //=============================================================================
  // explicit instantiations
  //=============================================================================
  template class MGTransferCUDA<2, double>;
  template class MGTransferCUDA<2, float>;
  template class MGTransferCUDA<3, double>;
  template class MGTransferCUDA<3, float>;

#define INSTANTIATE_COPY_TO_MG(dim, number_type, vec_number_type)            \
  template void MGTransferCUDA<dim, number_type>::copy_to_mg(                \
    const DoFHandler<dim> &,                                                 \
    MGLevelObject<                                                           \
      LinearAlgebra::distributed::Vector<number_type, MemorySpace::CUDA>> &, \
    const LinearAlgebra::distributed::Vector<vec_number_type,                \
                                             MemorySpace::CUDA> &) const

  INSTANTIATE_COPY_TO_MG(2, double, double);
  INSTANTIATE_COPY_TO_MG(2, float, float);
  INSTANTIATE_COPY_TO_MG(2, float, double);

  INSTANTIATE_COPY_TO_MG(3, double, double);
  INSTANTIATE_COPY_TO_MG(3, float, float);
  INSTANTIATE_COPY_TO_MG(3, float, double);


#define INSTANTIATE_COPY_FROM_MG(dim, number_type, vec_number_type)           \
  template void MGTransferCUDA<dim, number_type>::copy_from_mg(               \
    const DoFHandler<dim> &,                                                  \
    LinearAlgebra::distributed::Vector<vec_number_type, MemorySpace::CUDA> &, \
    const MGLevelObject<                                                      \
      LinearAlgebra::distributed::Vector<number_type, MemorySpace::CUDA>> &)  \
    const

  INSTANTIATE_COPY_FROM_MG(2, double, double);
  INSTANTIATE_COPY_FROM_MG(2, float, float);
  INSTANTIATE_COPY_FROM_MG(2, float, double);

  INSTANTIATE_COPY_FROM_MG(3, double, double);
  INSTANTIATE_COPY_FROM_MG(3, float, float);
  INSTANTIATE_COPY_FROM_MG(3, float, double);

#define INSTANTIATE_COPY_FROM_MG_ADD(dim, number_type, vec_number_type)       \
  template void MGTransferCUDA<dim, number_type>::copy_from_mg_add(           \
    const DoFHandler<dim> &,                                                  \
    LinearAlgebra::distributed::Vector<vec_number_type, MemorySpace::CUDA> &, \
    const MGLevelObject<                                                      \
      LinearAlgebra::distributed::Vector<number_type, MemorySpace::CUDA>> &)  \
    const

  INSTANTIATE_COPY_FROM_MG_ADD(2, double, double);
  INSTANTIATE_COPY_FROM_MG_ADD(2, float, float);
  INSTANTIATE_COPY_FROM_MG_ADD(2, float, double);

  INSTANTIATE_COPY_FROM_MG_ADD(3, double, double);
  INSTANTIATE_COPY_FROM_MG_ADD(3, float, float);
  INSTANTIATE_COPY_FROM_MG_ADD(3, float, double);

} // namespace PSMF

DEAL_II_NAMESPACE_OPEN


template class MGTransferBase<
  LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>;
template class MGTransferBase<
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>;

template class MGSmootherBase<
  LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>;
template class MGSmootherBase<
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>;

template class MGMatrixBase<
  LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>;
template class MGMatrixBase<
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>;

template class MGCoarseGridBase<
  LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>;
template class MGCoarseGridBase<
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>;

template class Multigrid<
  LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>;
template class Multigrid<
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>;


template <typename VectorType>
void
MGSmootherBase<VectorType>::apply(const unsigned int level,
                                  VectorType        &u,
                                  const VectorType  &rhs) const
{
  u = typename VectorType::value_type(0.);
  smooth(level, u, rhs);
}



template <typename VectorType>
void
MGTransferBase<VectorType>::prolongate_and_add(const unsigned int to_level,
                                               VectorType        &dst,
                                               const VectorType  &src) const
{
  VectorType temp;
  temp.reinit(dst, true);

  this->prolongate(to_level, temp, src);

  dst += temp;
}


DEAL_II_NAMESPACE_CLOSE