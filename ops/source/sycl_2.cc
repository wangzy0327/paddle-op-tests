#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_sign_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  float*  var_0 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space2_fn_sign_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 28; flat_i += 1) {
        cinn_sycl_store(var_0, IndexVec<4000>::Ramp((4000 * flat_i)), cinn_sycl_select((DataVec<float, 4000>::Load(x, (4000 * flat_i)) < 0.00000000f), -1.00000000f, cinn_sycl_select((DataVec<float, 4000>::Load(x, (4000 * flat_i)) > 0.00000000f), 1.00000000f, cinn_sycl_select((DataVec<float, 4000>::Load(x, (4000 * flat_i)) == 0.00000000f), 0.00000000f, DataVec<float, 4000>::Load(x, (4000 * flat_i))))));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
