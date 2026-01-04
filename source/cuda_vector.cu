/**
 * @file cuda_vector.cu
 * Created by Cu Cui on 2022/12/25.
 */

#include <deal.II/lac/read_write_vector.templates.h>

#include "cuda_vector.cuh"
#include "cuda_vector.template.cuh"

namespace PSMF
{

  template class CudaVector<unsigned int>;
  template class CudaVector<float>;
  template class CudaVector<double>;

} // namespace PSMF

DEAL_II_NAMESPACE_OPEN

template class LinearAlgebra::ReadWriteVector<unsigned int>;

DEAL_II_NAMESPACE_CLOSE