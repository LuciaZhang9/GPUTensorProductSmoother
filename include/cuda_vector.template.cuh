/**
 * @file cuda_vector.cuh
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef CUDA_VECTOR_TEMPLATE_CUH
#define CUDA_VECTOR_TEMPLATE_CUH

#include <deal.II/base/cuda.h>
#include <deal.II/base/cuda_size.h>

#include <deal.II/lac/cuda_kernels.h>
#include <deal.II/lac/read_write_vector.h>

#include "cuda_vector.cuh"

namespace PSMF
{

  using dealii::CUDAWrappers::block_size;
  using dealii::CUDAWrappers::chunk_size;

  template <typename Number>
  CudaVector<Number>::CudaVector(const CudaVector<Number> &V)
    : val(Utilities::CUDA::allocate_device_data<Number>(V.n_elements),
          Utilities::CUDA::delete_device_data<Number>)
    , n_elements(V.n_elements)
  {
    // Copy the values.
    const cudaError_t error_code = cudaMemcpy(val.get(),
                                              V.val.get(),
                                              n_elements * sizeof(Number),
                                              cudaMemcpyDeviceToDevice);
    AssertCuda(error_code);
  }

  template <typename Number>
  CudaVector<Number>::CudaVector()
    : val(nullptr, Utilities::CUDA::delete_device_data<Number>)
    , n_elements(0)
  {}

  template <typename Number>
  CudaVector<Number>::~CudaVector()
  {}

  template <typename Number>
  void
  CudaVector<Number>::reinit(const size_type n, const bool omit_zeroing_entries)
  {
    // Resize the underlying array if necessary
    if (n == 0)
      val.reset();
    else if (n != n_elements)
      val.reset(Utilities::CUDA::allocate_device_data<Number>(n));

    // If necessary set the elements to zero
    if (omit_zeroing_entries == false)
      {
        const cudaError_t error_code =
          cudaMemset(val.get(), 0, n * sizeof(Number));
        AssertCuda(error_code);
      }
    n_elements = n;
  }

  template <typename Number>
  void
  CudaVector<Number>::import(const LinearAlgebra::ReadWriteVector<Number> &V,
                             VectorOperation::values operation)
  {
    if (operation == VectorOperation::insert)
      {
        const cudaError_t error_code = cudaMemcpy(val.get(),
                                                  V.begin(),
                                                  n_elements * sizeof(Number),
                                                  cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }
    else if (operation == VectorOperation::add)
      {
        AssertThrow(false, ExcNotImplemented());
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }


  template <typename Number>
  std::size_t
  CudaVector<Number>::memory_consumption() const
  {
    std::size_t memory = sizeof(*this);
    memory += sizeof(Number) * static_cast<std::size_t>(n_elements);

    return memory;
  }

} // namespace PSMF

#endif // CUDA_VECTOR_TEMPLATE_CUH