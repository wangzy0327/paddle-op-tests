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

from op_test import OpTest, is_compile_with_device
from op_test_helper import TestCaseHelper, run_test

import paddle
# from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import numpy as np
import time

class TestSortOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info)) 
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.case["shape"], self.case["dtype"])}
        self.axis = self.case["axis"]
        self.descending = self.case["descending"]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)  
        x1 = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time()         
        out = paddle.sort(x1, self.axis, self.descending)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds") 
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("sort")
        x1 = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        print("CINN running at ", target.arch) 
        out = builder.sort(x1, self.axis, not self.descending)
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
        # forward_res = self.get_cinn_output(
        #     prog, target, [x1], [self.inputs["x"]], [out]
        # )

        # self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSortOpDumpicateElement(TestSortOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random([128], "int32", -10, 10)}
        self.axis = 0
        self.descending = False


# This test case will cause CINN to allocate a large amount of GPU memory, nearly 10 GB.
class TestSortOpLargeCudaMemoryOccupation(TestSortOp):
    def setUp(self):
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random([8192], "float32")}
        self.axis = 0
        self.descending = False


class TestSortOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpShapeTest"
        self.cls = TestSortOp
        self.inputs = [
            # {
            #     "shape": [512],
            # },
            {
                "shape": [1024],
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
            # TODO: known issue cinn/hlir/op/contrib/sort.cc:201
            # the array will exceed the cuda kernel stack size limit
            # {
            #     "shape": [32768],
            # },
            # {
            #     "shape": [65536],
            # },
            # {
            #     "shape": [131072],
            # },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axis": 0, "descending": False}]


class TestSortOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpDtypeTest"
        self.cls = TestSortOp
        self.inputs = [
            {
                "shape": [1024],
            },
            {
                "shape": [64, 16],
            },
            {
                "shape": [4, 32, 8],
            },
            {
                "shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            # {"dtype": "float64"},
            # {"dtype": "int32"},
            # {"dtype": "int64"},
        ]
        self.attrs = [{"axis": 0, "descending": False}]


class TestSortOpAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSortOpAttrsTest"
        self.cls = TestSortOp
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


class TestSortOpDescedingTest(TestSortOpShapeTest):
    def init_attrs(self):
        self.class_name = "TestSortOpDescedingTest"
        self.cls = TestSortOp
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
    run_test(TestSortOpDumpicateElement)
    # run_test(TestSortOpLargeCudaMemoryOccupation)

    TestSortOpShapeTest().run()
    TestSortOpDtypeTest().run()
    TestSortOpAxisTest().run()
    # TestSortOpDescedingTest().run()
