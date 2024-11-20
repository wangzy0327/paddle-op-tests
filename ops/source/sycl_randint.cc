#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_cast_1_fill_constant_2_mod_3_fill_constant_4_elementwise_add_5_6_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const int32_t*  var = (int32_t* )(*(void **)(void_args[0]));
  int32_t*  var_4 = (int32_t* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space0_fn_cast_1_fill_constant_2_mod_3_fill_constant_4_elementwise_add_5_6_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      cinn_sycl_store(var_4, IndexVec<24>::Ramp(0), cinn_sycl_mod_int32(DataVec<int32_t, 24>::Load(var, 0), 8));
    });
  });
}

#ifdef __cplusplus
}
#endif
