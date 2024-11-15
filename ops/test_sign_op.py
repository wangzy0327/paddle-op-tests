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
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestSignOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))         
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.case["shape"], self.case["dtype"], -10, 10)
        }

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)         
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time()          
        out = paddle.sign(x)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)

        print(f"Paddle Execution time: {execution_time:.6f} seconds")
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("sign")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        print("CINN running at ", target.arch)        
        out = builder.sign(x)

        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["x"],
        ]
        
        computation.get_tensor("x").from_numpy(tensor_data[0], target)
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
        # res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)


class TestSignOp1(TestSignOp):
    def init_case(self):
        self.inputs = {
            "x": np.array([1, 0, -1, np.nan, np.inf, -np.inf]).astype("float32")
        }


class TestSignOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSignOpShape"
        self.cls = TestSignOp
        self.inputs = [
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
            #     "shape": [80, 40, 5, 7],
            # },
            # {
            #     "shape": [80, 1, 5, 7],
            # },
            # {
            #     "shape": [80, 3, 32, 7],
            # },
            # {
            #     "shape": [10, 5, 32, 32],
            # },
            # {
            #     "shape": [1],
            # },
            # {
            #     "shape": [512],
            # },
            {
                "shape": [1024],
            },
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


class TestSignOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSignOpDtype"
        self.cls = TestSignOp
        self.inputs = [
            # {
            #     "shape": [1],
            # },
            {
                "shape": [5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            # {"dtype": "float16"},
            {"dtype": "float32"},
            # {"dtype": "float64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestSignOpShape().run()
    TestSignOpDtype().run()
