#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_slice_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  inputs = (float* )(*(void **)(void_args[0]));
  float*  var_0 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space1_fn_slice_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      cinn_sycl_store(var_0, IndexVec<25>::Ramp(0), DataVec<float, 25>::Load(inputs, (14 + (((IndexVec<25>::Ramp(0) / 5) * 12) + (2 * (IndexVec<25>::Ramp(0) % 5))))));
    });
  });
}

#ifdef __cplusplus
}
#endif
