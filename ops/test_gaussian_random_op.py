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
class TestGaussianRandomOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info)) 
        # print(f"\n{self.__class__.__name__}: {self.case}")
        pass

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)        
        # 记录开始时间
        start_time = time.time() 
        out = paddle.tensor.random.gaussian(
            shape=self.case["shape"],
            mean=self.case["mean"],
            std=self.case["std"],
            dtype=self.case["dtype"],
        )
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")        
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("gaussian_random")
        print("CINN running at ", target.arch)         
        out = builder.gaussian_random(
            self.case["shape"],
            self.case["mean"],
            self.case["std"],
            self.case["seed"],
            self.case["dtype"],
        )
        
        # computation = frontend.Computation.build_and_compile(target, builder)
        
        # # 记录开始时间
        # start_time = time.time()
        # computation.execute()
        # end_time = time.time()
        # # 计算执行时间
        # execution_time = end_time - start_time

        # print(f"CINN Execution time: {execution_time:.6f} seconds")
        # res_tensor = computation.get_tensor(str(out))
        # res_data = res_tensor.numpy(target)
        # # print(res_data)
        # output = paddle.to_tensor(res_data, stop_gradient=True)
        # # print(output)
        # self.cinn_outputs = [output]        
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        res_tensor = paddle.to_tensor(res)
        # res_tensor = paddle.to_tensor(res[0])
        # print(type(res_tensor))
        self.cinn_outputs = res_tensor

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Uniform distribution.
        self.check_outputs_and_grads(
            max_relative_error=10000, max_absolute_error=10000
        )


class TestGaussianRandomOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestGaussianRandomOpCase"
        self.cls = TestGaussianRandomOp
        self.inputs = [
            # {
            #     "shape": [1],
            # },
            {
                "shape": [1024],
            },
            # {
            #     "shape": [512, 256],
            # },
            # {
            #     "shape": [128, 64, 32],
            # },
            # {
            #     "shape": [16, 8, 4, 2],
            # },
            # {
            #     "shape": [16, 8, 4, 2, 1],
            # },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "mean": 0.0,
                "std": 0.0,
                "seed": 1234,
            },
        ]


class TestGaussianRandomOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestGaussianRandomOpCase"
        self.cls = TestGaussianRandomOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
            # {
            #     "dtype": "float64",
            # },
        ]
        self.attrs = [
            {
                "mean": 0.0,
                "std": 0.0,
                "seed": 1234,
            },
        ]


class TestGaussianRandomOpAttr(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestGaussianRandomOpCase"
        self.cls = TestGaussianRandomOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {
                "mean": 1.0,
                "std": 0.0,
                "seed": 1,
            },
            {
                "mean": 0.0,
                "std": 1.0,
                "seed": 2,
            },
            {
                "mean": 1.0,
                "std": 1.0,
                "seed": 3,
            },
        ]


if __name__ == "__main__":
    TestGaussianRandomOpShape().run()
    TestGaussianRandomOpDtype().run()
    TestGaussianRandomOpAttr().run()
