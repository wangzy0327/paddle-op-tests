#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_isclose_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  const float*  y = (float* )(*(void **)(void_args[1]));
  bool*  var_1 = (bool* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space4_fn_isclose_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      for (int32_t flat_i = 0; flat_i < 1024; flat_i += 1) {
        cinn_sycl_store(var_1, flat_i, cinn_sycl_select((cinn_sycl_isnan_fp32(x[flat_i]) || cinn_sycl_isnan_fp32(y[flat_i])), (false && (cinn_sycl_isnan_fp32(x[flat_i]) == cinn_sycl_isnan_fp32(y[flat_i]))), (((x[flat_i] == y[flat_i]) || (cinn_sycl_select((x[flat_i] > y[flat_i]), (x[flat_i] - y[flat_i]), (y[flat_i] - x[flat_i])) <= (9.99999994e-09f + cinn_sycl_select((y[flat_i] > 0.00000000f), (9.99999994e-09f * y[flat_i]), (-9.99999994e-09f * y[flat_i]))))) || (cinn_sycl_select((cinn_sycl_select((x[flat_i] > y[flat_i]), (x[flat_i] - y[flat_i]), (y[flat_i] - x[flat_i])) > (9.99999994e-09f + cinn_sycl_select((y[flat_i] > 0.00000000f), (9.99999994e-09f * y[flat_i]), (-9.99999994e-09f * y[flat_i])))), (-9.99999994e-09f + (cinn_sycl_select((x[flat_i] > y[flat_i]), (x[flat_i] - y[flat_i]), (y[flat_i] - x[flat_i])) - cinn_sycl_select((y[flat_i] > 0.00000000f), (9.99999994e-09f * y[flat_i]), (-9.99999994e-09f * y[flat_i])))), (9.99999994e-09f + (cinn_sycl_select((y[flat_i] > 0.00000000f), (9.99999994e-09f * y[flat_i]), (-9.99999994e-09f * y[flat_i])) - cinn_sycl_select((x[flat_i] > y[flat_i]), (x[flat_i] - y[flat_i]), (y[flat_i] - x[flat_i]))))) <= 1.19209290e-07f))));
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
