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
class TestTransposeOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.case["shape"], self.case["dtype"])}
        self.axes = self.case["axes"]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)         
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time()  
        out = paddle.transpose(
            x,
            [
                axis + len(self.inputs["x"].shape) if axis < 0 else axis
                for axis in self.axes
            ],
        )
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("transpose_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        print("CINN running at ", target.arch)   
        out = builder.transpose(x, self.axes)

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
        output = paddle.to_tensor(res_data, stop_gradient=True)
        # print(output)
        self.cinn_outputs = [output]
        # prog = builder.build()
        # res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTransposeOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTransposeOpShapeTest"
        self.cls = TestTransposeOp
        self.inputs = [
            {"shape": [512], "axes": [0]},
            # {"shape": [1024], "axes": [-1]},
            # {"shape": [1200], "axes": [0]},
            # {"shape": [64, 16], "axes": [1, 0]},
            # {"shape": [4, 32, 8], "axes": [1, 0, 2]},
            # {"shape": [16, 8, 4, 2], "axes": [0, 2, 1, 3]},
            # {"shape": [2, 8, 4, 2, 5], "axes": [0, 2, 3, 1, 4]},
            # {"shape": [4, 8, 1, 2, 16], "axes": [0, 2, 4, 1, 3]},
            # {"shape": [1], "axes": [0]},
            # {"shape": [1, 1, 1, 1], "axes": [0, 2, 3, 1]},
            # {"shape": [1, 1, 1, 1, 1], "axes": [0, 2, 3, 1, 4]},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestTransposeOpOnesTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTransposeOpOnesTest"
        self.cls = TestTransposeOp
        self.inputs = [
            {"shape": [1], "axes": [0]},
            {"shape": [1, 1, 1, 1], "axes": [0, 2, 3, 1]},
            {"shape": [1, 1, 1, 1, 1], "axes": [0, 2, 3, 1, 4]},
            {"shape": [1, 1, 512, 1, 1], "axes": [0, 2, 3, 1, 4]},
            {"shape": [1, 2048, 1, 1], "axes": [0, 2, 3, 1]},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestTransposeOpLargeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTransposeOpLargeTest"
        self.cls = TestTransposeOp
        self.inputs = [
            {"shape": [2048], "axes": [0]},
            {"shape": [1, 1, 65536, 1], "axes": [0, 2, 3, 1]},
            {"shape": [1, 1, 131072, 1, 1], "axes": [0, 2, 3, 1, 4]},
            {"shape": [1, 1048576, 1, 1], "axes": [0, 2, 3, 1]},
            {"shape": [16, 32, 64, 32], "axes": [0, 2, 3, 1]},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestTransposeOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTransposeOpDtypeTest"
        self.cls = TestTransposeOp
        self.inputs = [
            {"shape": [1024], "axes": [0]},
            {"shape": [64, 16], "axes": [1, 0]},
            {"shape": [4, 32, 8], "axes": [0, 2, 1]},
            {"shape": [16, 8, 4, 2], "axes": [1, 2, 3, 0]},
        ]
        self.dtypes = [
            # {"dtype": "float16"},
            {"dtype": "float32"},
            # {"dtype": "float64"},
            # {"dtype": "bool"},
            # {"dtype": "int32"},
            # {"dtype": "int64"},
        ]
        self.attrs = []


class TestTransposeOpAttributeAxes(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestTransposeOpAttributeAxes"
        self.cls = TestTransposeOp
        self.inputs = [
            {"shape": [1024], "axes": [0]},
            {"shape": [1024], "axes": [-1]},
            {"shape": [64, 16], "axes": [1, 0]},
            {"shape": [64, 16], "axes": [0, -1]},
            {"shape": [4, 32, 8], "axes": [0, 2, 1]},
            {"shape": [4, 32, 8], "axes": [-3, -1, 1]},
            {"shape": [16, 8, 4, 2], "axes": [1, 2, 3, 0]},
            {"shape": [16, 8, 4, 2], "axes": [1, -2, -1, -4]},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestTransposeOpShapeTest().run()
    # TestTransposeOpOnesTest().run()
    # TestTransposeOpLargeTest().run()
    TestTransposeOpDtypeTest().run()
    # TestTransposeOpAttributeAxes().run()
