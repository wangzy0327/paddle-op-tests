#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from op_test import OpTest, OpTestTool, is_compile_with_device
from op_test_helper import TestCaseHelper

import paddle
# from paddle.cinn.common import is_compiled_with_cuda
# from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import numpy as np
import time


@OpTestTool.skip_if(
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestConstantOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))        
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.name = "x"
        dtype = self.case["dtype"]
        if "constant_value" in self.case:
            if "bool" in dtype:
                self.value = bool(self.case["constant_value"])
            elif "int" in dtype:
                self.value = int(self.case["constant_value"])
            elif "float" in dtype:
                self.value = float(self.case["constant_value"])
        else:
            self.value = self.random(self.case["shape"], dtype).tolist()
        self.dtype = dtype

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)    
        # 记录开始时间
        start_time = time.time()            
        x = paddle.to_tensor(self.value, dtype=self.dtype)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        print(x)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")        
        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("constant")
        print("CINN running at ", target.arch)    
        x = builder.constant(self.value, self.name, self.dtype)
        # computation = frontend.Computation.build_and_compile(target, builder)
        
        # tensor_data = [
        #     x,
        # ]
        
        # computation.get_tensor("x").from_numpy(tensor_data[0], target)
        # # 记录开始时间
        # start_time = time.time()
        # computation.execute()
        # end_time = time.time()
        # # 计算执行时间
        # execution_time = end_time - start_time

        # print(f"CINN Execution time: {execution_time:.6f} seconds")
        # res_tensor = computation.get_tensor(str(x))
        # res_data = res_tensor.numpy(target)
        # # print(res_data)
        # output = paddle.to_tensor(res_data, stop_gradient=True)
        # print(output)
        # self.cinn_outputs = [output]            
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestConstantOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConstantOpShape"
        self.cls = TestConstantOp
        self.inputs = [
            {
                "constant_value": 10,
            },
            {
                "constant_value": -5,
            },
            # {
            #     "shape": [10],
            # },
            # {
            #     "shape": [8, 5],
            # },
            # {
            #     "shape": [10, 3, 5],
            # },
            # {
            #     "shape": [1, 2, 4, 8],
            # },
            # # known issue: https://github.com/PaddlePaddle/CINN/pull/1453
            # # The compilation time is particularly long for AssignValue op.
            # # {
            # #     "shape": [16, 4, 8, 32],
            # # },
            # {
            #     "shape": [1],
            # },
            # {
            #     "shape": [512],
            # },
            {
                "shape": [1024],
            },
            # Update: stack over flow while compiling
            # very slow for the shape 2048
            # {
            #     "shape": [2048],
            # },
            # {
            #     "shape": [1, 1, 1, 1],
            # },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestConstantOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestConstantOpDtype"
        self.cls = TestConstantOp
        self.inputs = [
            {
                "constant_value": 1,
            },
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
        ]
        self.dtypes = [
            # {"dtype": "float16"},
            {"dtype": "float32"},
            # {"dtype": "float64"},
            # {"dtype": "bool"},
            # {"dtype": "uint8"},
            # {"dtype": "int8"},
            # {"dtype": "int32"},
            # {"dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestConstantOpShape().run()
    TestConstantOpDtype().run()
