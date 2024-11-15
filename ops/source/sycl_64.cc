#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_bitwise_and_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const int16_t*  x = (int16_t* )(*(void **)(void_args[0]));
  const int16_t*  y = (int16_t* )(*(void **)(void_args[1]));
  int16_t*  var_1 = (int16_t* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space64_fn_bitwise_and_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 16; flat_i += 1) {
        cinn_sycl_store(var_1, flat_i, cinn_sycl_bitwise_and_int16(x[flat_i], y[flat_i]));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
