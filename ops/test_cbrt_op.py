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
class TestCbrtOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(
                self.case["shape"], self.case["dtype"], 0.0, 100.0
            ),
            "y": np.full(self.case["shape"], 0.333333, self.case["dtype"])
        }
        self.axis = np.random.choice([-1, 0])

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)    
        # 记录开始时间
        start_time = time.time() 
        # print(self.inputs["x"])           
        # print(self.inputs["y"])           
        numpy_out = np.cbrt(self.inputs["x"])
        out = paddle.to_tensor(numpy_out, stop_gradient=False)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time 
        # print(out)    
        print(f"Paddle Execution time: {execution_time:.6f} seconds")             
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        # builder = frontend.NetBuilder("cbrt")
        builder = frontend.NetBuilder("pow")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )        
        print("CINN running at ", target.arch)        
        out = builder.pow(x,y,axis=self.axis)
        
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["x"],
            self.inputs["y"]
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
        # res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(
            max_relative_error=1e-3, max_absolute_error=1e-3
        )


class TestCbrtOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCbrtOpShape"
        self.cls = TestCbrtOp
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
            #     "shape": [80, 3, 1024, 7],
            # },
            # {
            #     "shape": [10, 5, 1024, 2048],
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


class TestCbrtOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestCbrtOpDtype"
        self.cls = TestCbrtOp
        self.inputs = [
            # {
            #     "shape": [1],
            # },
            # {
            #     "shape": [5],
            # },
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
    TestCbrtOpShape().run()
    TestCbrtOpDtype().run()
