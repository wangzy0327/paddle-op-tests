#!/bin/bash

# 定义 Python 解释器的路径
python_path="python3"

# 设置环境变量
# export FLAGS_cinn_use_cuda_vectorize=1

# 定义要运行的 Python 脚本列表
scripts=(
    "test_abs_op.py"
    "test_acos_op.py"
    "test_acosh_op.py"
    "test_add_op.py" 
    "test_arange_op.py" 
    "test_argmax_op.py"
    "test_argmin_op.py"
    "test_argsort_op.py" 
    "test_asin_op.py" 
    "test_asinh_op.py" 
    "test_atan_op.py" 
    "test_atanh_op.py"
    "test_batch_norm_op.py" 
    "test_binary_elementwise_op.py" 
    "test_bitcast_convert_op.py" 
    "test_bitwise_op.py" 
    "test_broadcast_to_op.py" 
    "test_cast_op.py" 
    "test_cbrt_op.py" 
    "test_ceil_op.py"
    "test_comparison_op.py" 
    "test_concat_op.py" "test_constant_op.py"
    "test_depthwise_conv2d_cinn.py" 
    "test_conv2d_cinn.py" 
    "test_cos_op.py"
    "test_cosh_op.py" 
    "test_divide_op.py"
    "test_dropout_infer_op.py" 
    "test_erf_op.py" "test_exp_op.py" 
    "test_expand_dims.py" 
    "test_floor_op.py" 
    "test_gather_op.py"
     "test_gaussian_random_op.py" "test_gelu_op.py" 
    "test_is_finite_op.py" 
    "test_is_inf_op.py" "test_is_nan_op.py" 
    "test_isclose_op.py" 
    "test_left_shift_op.py"
     "test_log_op.py" 
    "test_matmul_op.py" 
    "test_max_op.py" 
    "test_min_op.py" 
    "test_mod_op.py" 
    "test_mul_op.py" 
    "test_multiply_op.py" 
    "test_negative_op.py" 
    "test_one_hot_op.py" "test_pool2d_op.py" 
    "test_pow_op.py"
     "test_randint_op.py" "test_reduce_op.py" 
    "test_relu_op.py"
     "test_repeat_op.py" 
    "test_reshape_op.py"
     "test_reverse_op.py" 
    "test_right_shift_op.py"
     "test_round_op.py" 
    "test_rsqrt_op.py"
    "test_select_op.py"
     "test_sigmoid_op.py" 
    "test_sign_op.py"
     "test_sin_op.py" "test_sinh_op.py" 
    "test_slice_op.py" 
    "test_softmax_op.py"
     "test_split_op.py" 
    "test_sqrt_op.py" 
    "test_squeeze_op.py"
     "test_subtract_op.py"  
    "test_trunc_op.py"
     "test_uniform_random_op.py"
    "test_sort_op.py" "test_sum_op.py"
    "test_tan_op.py" "test_tanh_op.py" "test_top_k_op.py" 
    "test_transpose_op.py" 
    "test_fill_constant_op.py"
     "test_trunc_op.py"  
)

# 日志文件
log_file="test_op2.log"
# 清空日志文件
> "$log_file"

# 统计变量
total_tests=${#scripts[@]}
passed_tests=0
failed_tests=0
declare -a failed_list
start_time=$(date +%s)

# 打印表头
echo "-------------------------------------------------------------"
printf "| %-5s | %-30s | %-8s | %-9s |\n" "ID" "Test Name" "Result" 
echo "-------------------------------------------------------------"

# 循环遍历并执行每个脚本
for ((i=0; i<${#scripts[@]}; i++)); do
    script=${scripts[$i]}
    test_name=$(basename "$script" .py)
    test_num=$((i+1))
    
    # 记录开始时间
    test_start=$(date +%s.%N)
    
    # 执行测试并捕获退出状态
    echo "Running $script..." | tee -a "$log_file"
    # 判断是否为指定测试脚本，决定是否设置 FLAGS_cinn_compile_level=1
    if [[ "$script" == "test_pow_op.py" || "$script" == "test_cbrt_op.py" ]]; then
        # 设置环境变量
        export FLAGS_cinn_compile_level=1
    else
        # 取消设置环境变量（如果之前有设置）
        unset FLAGS_cinn_compile_level
    fi    
    if $python_path "$script" &>> "$log_file"; then
        result="Passed"
        ((passed_tests++))
    else
        result="Failed"
        ((failed_tests++))
        failed_list+=("$script")
    fi
    
    # 计算执行时间
    # test_end=$(date +%s.%N)
    # test_time=$(echo "$test_end - $test_start" | bc | awk '{printf "%.2f", $0}')
    
    # 打印当前测试结果
    printf "| %3d/%-3d | %-30s | %-8s |\n" \
           "$test_num" "$total_tests" "$test_name" "$result" | tee -a "$log_file"
done

# 计算总时间
end_time=$(date +%s)
total_time=$((end_time - start_time))

# 打印汇总统计
echo "-------------------------------------------------------------"
echo "Summary:"
echo "  Total tests:   $total_tests"
echo "  Passed:        $passed_tests"
echo "  Failed:        $failed_tests"

# 打印失败列表（如果有）
if [ "$failed_tests" -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for failed_test in "${failed_list[@]}"; do
        echo "  - $failed_test"
    done
fi

# 最终结果
echo ""
if [ "$failed_tests" -eq 0 ]; then
    echo "SUCCESS: All tests passed!"
    exit 0
else
    echo "ERROR: $failed_tests tests failed!"
    exit 1
fi
