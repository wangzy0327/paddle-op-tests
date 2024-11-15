#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_select_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const bool*  Condition = (bool* )(*(void **)(void_args[0]));
  const float*  X = (float* )(*(void **)(void_args[1]));
  const float*  Y = (float* )(*(void **)(void_args[2]));
  float*  var_2 = (float* )(*(void **)(void_args[3]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space0_fn_select_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      cinn_sycl_store(var_2, IndexVec<1024>::Ramp(0), cinn_sycl_select(DataVec<bool, 1024>::Load(Condition, 0), DataVec<float, 1024>::Load(X, 0), DataVec<float, 1024>::Load(Y, 0)));
    });
  });
}

#ifdef __cplusplus
}
#endif
