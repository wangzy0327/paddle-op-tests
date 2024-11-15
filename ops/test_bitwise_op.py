# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import time
import numpy as np

@OpTestTool.skip_if(
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestBitwiseOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        # Test with infinite values
        if "with_inf" in self.case:
            self.x_np = np.full(
                shape=self.case["x_shape"],
                fill_value=np.inf,
                dtype=self.case["dtype"],
            )
        # Test with nan values
        elif "with_nan" in self.case:
            self.x_np = np.full(
                shape=self.case["x_shape"],
                fill_value=np.nan,
                dtype=self.case["dtype"],
            )
        else:
            self.x_np = self.random(
                shape=self.case["x_shape"], dtype=self.case["dtype"]
            )
        if self.case["op_type"] != "not":
            self.y_np = self.random(
                shape=self.case["y_shape"], dtype=self.case["dtype"]
            )

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)          
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        # 记录开始时间
        start_time = time.time()          
        if self.case["op_type"] != "not":
            y = paddle.to_tensor(self.y_np, stop_gradient=False)
        if self.case["op_type"] == "and":
            out = paddle.bitwise_and(x, y)
        elif self.case["op_type"] == "or":
            out = paddle.bitwise_or(x, y)
        elif self.case["op_type"] == "xor":
            out = paddle.bitwise_xor(x, y)
        elif self.case["op_type"] == "not":
            out = paddle.bitwise_not(x)
        else:
            out = paddle.assign(x)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time   
        print(f"Paddle Execution time: {execution_time:.6f} seconds")                  
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("bitwise")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["x_shape"], "x"
        )
        print("CINN running at ", target.arch)         
        if self.case["op_type"] != "not":
            y = builder.create_input(
                self.nptype2cinntype(self.case["dtype"]),
                self.case["y_shape"],
                "y",
            )
        if self.case["op_type"] == "and":
            out = builder.bitwise_and(x, y)
        elif self.case["op_type"] == "or":
            out = builder.bitwise_or(x, y)
        elif self.case["op_type"] == "xor":
            out = builder.bitwise_xor(x, y)
        elif self.case["op_type"] == "not":
            out = builder.bitwise_not(x)
        else:
            out = builder.identity(x)
            
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = []
        if self.case["op_type"] != "not":        
            tensor_data = [
                self.x_np,
                self.y_np
            ]
        else:
            tensor_data = [
                self.x_np
            ]
        
        computation.get_tensor("x").from_numpy(tensor_data[0], target)
        if self.case["op_type"] != "not":        
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
        # if self.case["op_type"] != "not":
        #     res = self.get_cinn_output(
        #         prog, target, [x, y], [self.x_np, self.y_np], [out]
        #     )
        # else:
        #     res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        # self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestBitwiseOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBitwiseOpCase"
        self.cls = TestBitwiseOp
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
            # {
            #     "x_shape": [1, 1, 1, 1, 1],
            #     "y_shape": [1, 1, 1, 1, 1],
            # },
        ]
        self.dtypes = [
            {"dtype": "int32"},
        ]
        self.attrs = [
            {"op_type": "and"},
            {"op_type": "or"},
            {"op_type": "xor"},
            {"op_type": "not"},
        ]


class TestBitwiseOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBitwiseOpCase"
        self.cls = TestBitwiseOp
        self.inputs = [
            {
                "x_shape": [32, 64],
                "y_shape": [32, 64],
            },
        ]
        self.dtypes = [
            # {"dtype": "bool"},
            # {"dtype": "uint8"},
            # {"dtype": "int8"},
            # {"dtype": "int16"},
            {"dtype": "int32"},
            # {"dtype": "int64"},
        ]
        self.attrs = [
            {"op_type": "and"},
            {"op_type": "or"},
            {"op_type": "xor"},
            {"op_type": "not"},
        ]


class TestBitwiseOpBroadcast(TestBitwiseOpShape):
    def init_attrs(self):
        super().init_attrs()
        self.inputs = [
            {
                "x_shape": [1024],
                "y_shape": [1],
            },
            {
                "x_shape": [512, 256],
                "y_shape": [1, 1],
            },
            # {
            #     "x_shape": [128, 64, 32],
            #     "y_shape": [1, 1, 1],
            # },
            # {
            #     "x_shape": [16, 8, 4, 2],
            #     "y_shape": [1, 1, 1, 1],
            # },
            # {
            #     "x_shape": [16, 8, 4, 2, 1],
            #     "y_shape": [1, 1, 1, 1, 1],
            # },
        ]


class TestBitwiseWithINF(TestBitwiseOpDtype):
    def init_attrs(self):
        super().init_attrs()
        self.inputs = [
            {
                "x_shape": [16],
                "y_shape": [16],
                "with_inf": True,
            },
        ]


class TestBitwiseWithNAN(TestBitwiseOpDtype):
    def init_attrs(self):
        super().init_attrs()
        self.inputs = [
            {
                "x_shape": [16],
                "y_shape": [16],
                "with_nan": True,
            },
        ]


if __name__ == "__main__":
    TestBitwiseOpShape().run()
    TestBitwiseOpDtype().run()
    TestBitwiseOpBroadcast().run()
    TestBitwiseWithINF().run()
    TestBitwiseWithNAN().run()
