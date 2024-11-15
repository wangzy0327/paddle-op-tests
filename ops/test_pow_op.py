#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest, OpTestTool, is_compile_with_device
from op_test_helper import TestCaseHelper

import paddle
# from paddle.cinn.common import is_compiled_with_cuda
# from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import time
import numpy as np


@OpTestTool.skip_if(
    # not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestPowOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))         
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["dtype"],
            low=self.case["base_low"],
            high=self.case["base_high"],
        )
        self.y_np = self.random(
            shape=self.case["y_shape"],
            dtype=self.case["dtype"],
            low=self.case["exp_low"],
            high=self.case["exp_high"],
        )
        self.axis = np.random.choice([-1, 0])

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)         
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        # 记录开始时间
        start_time = time.time()         
        out = paddle.pow(x, y)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(y)

        print(f"Paddle Execution time: {execution_time:.6f} seconds")          
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("pow")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        y = builder.create_input(
            self.nptype2cinntype(self.y_np.dtype), self.y_np.shape, "y"
        )
        print("CINN running at ", target.arch)         
        out = builder.pow(x, y, axis=self.axis)
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.x_np,
            self.y_np,
        ]
        
        computation.get_tensor("x").from_numpy(tensor_data[0], target)
        computation.get_tensor("y").from_numpy(tensor_data[1], target)
        # 记录开始时间
        start_time = time.time()
        computation.execute()
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time

        print(f"CINN Execution time: {execution_time:.6f} seconds")
        res_tensor = computation.get_tensor(str(out))
        res_data = res_tensor.numpy(target)
        # print(res_data)
        output = paddle.to_tensor(res_data, stop_gradient=False)
        # print(output)
        self.cinn_outputs = [output]        
        # prog = builder.build()
        # res = self.get_cinn_output(
        #     prog, target, [x, y], [self.x_np, self.y_np], [out]
        # )
        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


class TestPowOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalRightShiftCase"
        self.cls = TestPowOp
        self.inputs = [
            # {
            #     "x_shape": [1],
            #     "y_shape": [1],
            # },
            {
                "x_shape": [1024],
                "y_shape": [1024],
            },
            # {
            #     "x_shape": [512, 256],
            #     "y_shape": [512, 256],
            # },
            # {
            #     "x_shape": [128, 64, 32],
            #     "y_shape": [128, 64, 32],
            # },
            # {
            #     "x_shape": [16, 8, 4, 2],
            #     "y_shape": [16, 8, 4, 2],
            # },
            # {
            #     "x_shape": [16, 8, 4, 2, 1],
            #     "y_shape": [16, 8, 4, 2, 1],
            # },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "base_low": -10,
                "base_high": 10,
                "exp_low": -3,
                "exp_high": 3,
            },
        ]


class TestPowOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalRightShiftCase"
        self.cls = TestPowOp
        self.inputs = [
            {
                "x_shape": [1024],
                "y_shape": [1024],
            },
        ]
        self.dtypes = [
            # {
            #     "dtype": "int32",
            # },
            # {
            #     "dtype": "int64",
            # },
            # {
            #     "dtype": "float16",
            # },
            {
                "dtype": "float32",
            },
            # {
            #     "dtype": "float64",
            # },
        ]
        self.attrs = [
            {
                "base_low": -10,
                "base_high": 10,
                "exp_low": -3,
                "exp_high": 3,
            },
        ]


if __name__ == "__main__":
    TestPowOpShape().run()
    TestPowOpDtype().run()
