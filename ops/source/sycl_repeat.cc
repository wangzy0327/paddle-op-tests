#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_repeat_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  float*  var_0 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space0_fn_repeat_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      cinn_sycl_store(var_0, IndexVec<2048>::Ramp(0), DataVec<float, 2048>::Load(x, cinn_sycl_select((((IndexVec<2048>::Ramp(0) > 0) && (2 > 0)) || ((IndexVec<2048>::Ramp(0) < 0) && (2 < 0))), (IndexVec<2048>::Ramp(0) / 2), cinn_sycl_select(((IndexVec<2048>::Ramp(0) % 2) == 0), (IndexVec<2048>::Ramp(0) / 2), ((IndexVec<2048>::Ramp(0) / 2) + -1)))));
    });
  });
}

#ifdef __cplusplus
}
#endif
