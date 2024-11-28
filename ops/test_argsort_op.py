#!/usr/bin/env python3

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
# from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import numpy as np
import time


@OpTestTool.skip_if(
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestArgSortOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info)) 
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.case["shape"], self.case["dtype"])
        self.axis = self.case["axis"]
        self.descending = self.case["descending"]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)         
        x1 = paddle.to_tensor(self.x_np , stop_gradient=True)
        # print("Paddle elements : ", self.x_np)
        # 记录开始时间
        start_time = time.time()        
        out = paddle.argsort(x1, self.axis, self.descending)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time 
        # from argsort return tensor(int64) to tensor(int32)
        out = out.cast('int32')
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")                
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("argsort")
        x1 = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x1"
        )
        # print("CINN elements : ", self.x_np)
        print("CINN running at ", target.arch)          
        out = builder.argsort(x1, self.axis, not self.descending)
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.x_np,
        ]
        
        computation.get_tensor("x1").from_numpy(tensor_data[0], target)
        # 记录开始时间
        start_time = time.time()
        computation.execute()
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time

        print(f"CINN Execution time: {execution_time:.6f} seconds")
        res_tensor = computation.get_tensor(str(out[0]))
        res_data = res_tensor.numpy(target)
        # print(res_data)
        output = paddle.to_tensor(res_data, stop_gradient=True)
        # print(output)
        self.cinn_outputs = [output]        
        # prog = builder.build()
        # forward_res = self.get_cinn_output(prog, target, [x1], [self.x_np], out)
        # print(forward_res[0])
        # self.cinn_outputs = paddle.to_tensor([forward_res[0]], stop_gradient=True)
        # self.cinn_outputs = self.cinn_outputs.cast("int32")
        # print(self.cinn_outputs)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgSortOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpShapeTest"
        self.cls = TestArgSortOp
        self.inputs = [
            # {
            #     "shape": [512],
            # },
            {
                "shape": [32],
            },
            # {
            #     "shape": [1200],
            # },
            # {
            #     "shape": [64, 16],
            # },
            # {
            #     "shape": [4, 32, 8],
            # },
            # {
            #     "shape": [16, 8, 4, 2],
            # },
            # {
            #     "shape": [2, 8, 4, 2, 5],
            # },
            # {
            #     "shape": [4, 8, 1, 2, 16],
            # },
            # {
            #     "shape": [1],
            # },
            # {
            #     "shape": [1, 1, 1, 1],
            # },
            # {
            #     "shape": [1, 1, 1, 1, 1],
            # },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axis": 0, "descending": False}]


class TestArgSortOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpDtypeTest"
        self.cls = TestArgSortOp
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
            # # Throw dtype not support error in paddle
            # # {
            # #     "dtype": "uint8",
            # # },
            # {
            #     "dtype": "int32",
            # },
            # {
            #     "dtype": "int64",
            # },
        ]
        self.attrs = [{"axis": 0, "descending": False}]


class TestArgSortOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpAxisTest"
        self.cls = TestArgSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "descending": False},
            {"axis": 1, "descending": False},
            {"axis": 2, "descending": False},
            {"axis": 3, "descending": False},
        ]


class TestArgSortOpDescendingTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "ArgSortOpDescendingTest"
        self.cls = TestArgSortOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [
            {"axis": 0, "descending": True},
            {"axis": 1, "descending": True},
            {"axis": 2, "descending": True},
            {"axis": 3, "descending": True},
        ]


if __name__ == "__main__":
    TestArgSortOpShapeTest().run()
    TestArgSortOpDtypeTest().run()
    TestArgSortOpAxisTest().run()
    TestArgSortOpDescendingTest().run()
