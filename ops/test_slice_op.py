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
class TestSliceOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))          
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "inputs": self.random(self.case["shape"], self.case["dtype"])
        }
        self.axes = self.case["axes"]
        self.starts = self.case["starts"]
        self.ends = self.case["ends"]
        self.strides = self.case["strides"]
        self.decrease_axis = self.case["decrease_axis"]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)          
        x = paddle.to_tensor(self.inputs["inputs"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time() 
        res = paddle.strided_slice(
            x, self.axes, self.starts, self.ends, self.strides
        )
        out_shape = []
        for i in range(len(res.shape)):
            if i in self.decrease_axis:
                self.assertEqual(res.shape[i], 1)
            else:
                out_shape.append(res.shape[i])

        if len(out_shape) == 0:
            out_shape = [1]
        res = paddle.reshape(res, out_shape)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")        
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("slice")
        inputs = builder.create_input(
            self.nptype2cinntype(self.inputs["inputs"].dtype),
            self.inputs["inputs"].shape,
            "inputs",
        )
        print("CINN running at ", target.arch)         
        out = builder.slice(
            inputs,
            axes=self.axes,
            starts=self.starts,
            ends=self.ends,
            strides=self.strides,
            decrease_axis=self.decrease_axis,
        )

        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["inputs"],
        ]
        
        computation.get_tensor("inputs").from_numpy(tensor_data[0], target)
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
        # res = self.get_cinn_output(
        #     prog, target, [inputs], [self.inputs["inputs"]], [out]
        # )
        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSliceOpLegacyTestCase(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpLegacyTestCase"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [2, 2],
                "ends": [5, 5],
                "strides": [1, 1],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [1, 2],
                "ends": [6, 1000],
                "strides": [1, 2],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [2, 1],
                "ends": [-1, 7],
                "strides": [3, 2],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [2, 1000],
                "ends": [8, 1],
                "strides": [1, -2],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [-1, -2],
                "ends": [-5, -8],
                "strides": [-1, -2],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [2, 2],
                "ends": [5, 3],
                "strides": [1, 1],
                "decrease_axis": [1],
            },
            {
                "shape": [10, 12],
                "axes": [0, 1],
                "starts": [2, 2],
                "ends": [3, 3],
                "strides": [1, 1],
                "decrease_axis": [0, 1],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSliceOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpShapeTest"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [64],
                "axes": [0],
                "starts": [2],
                "ends": [5],
                "strides": [1],
                "decrease_axis": [],
            },
            {
                "shape": [128, 32],
                "axes": [0, 1],
                "starts": [24, 10],
                "ends": [56, 26],
                "strides": [1, 1],
                "decrease_axis": [],
            },
            {
                "shape": [32, 10, 64],
                "axes": [0, 1, 2],
                "starts": [24, 4, 0],
                "ends": [32, 8, 64],
                "strides": [1, 1, 4],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12, 9, 5],
                "axes": [0, 1, 2],
                "starts": [2, 4, 0],
                "ends": [5, 9, 7],
                "strides": [1, 1, 2],
                "decrease_axis": [],
            },
            {
                "shape": [1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
                "decrease_axis": [],
            },
            {
                "shape": [1, 1, 1, 1, 1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
                "decrease_axis": [],
            },
            {
                "shape": [1024, 1, 2],
                "axes": [0],
                "starts": [128],
                "ends": [640],
                "strides": [1],
                "decrease_axis": [],
            },
            {
                "shape": [2, 4096, 8],
                "axes": [1],
                "starts": [1024],
                "ends": [3072],
                "strides": [1],
                "decrease_axis": [],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSliceOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpDtypeTest"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [9, 5, 4, 7],
                "axes": [0, 1, 3],
                "starts": [2, 2, 0],
                "ends": [5, 5, 6],
                "strides": [1, 2, 4],
                "decrease_axis": [],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
            # {"dtype": "float64"},
            # {"dtype": "int32"},
            # {"dtype": "int64"},
            # {"dtype": "bool"},
        ]
        self.attrs = []


class TestSliceOpAxesTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpAxesTest"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [128, 32],
                "axes": [1],
                "starts": [10],
                "ends": [26],
                "strides": [1],
                "decrease_axis": [],
            },
            {
                "shape": [32, 10, 64],
                "axes": [0, 2],
                "starts": [24, 0],
                "ends": [32, 64],
                "strides": [1, 4],
                "decrease_axis": [],
            },
            {
                "shape": [10, 12, 9, 5],
                "axes": [0, 3],
                "starts": [2, 0],
                "ends": [5, 3],
                "strides": [1, 1],
                "decrease_axis": [],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSliceOpStridesTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpStridesTest"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [128, 32],
                "axes": [0, 1],
                "starts": [0, 0],
                "ends": [128, 32],
                "strides": [16, 2],
                "decrease_axis": [],
            },
            {
                "shape": [32, 10, 64],
                "axes": [0, 2],
                "starts": [16, 0],
                "ends": [32, 64],
                "strides": [2, 4],
                "decrease_axis": [],
            },
            {
                "shape": [8, 16, 32, 64, 128],
                "axes": [0, 1, 2, 3, 4],
                "starts": [0, 0, 0, 0, 0],
                "ends": [8, 16, 32, 64, 128],
                "strides": [1, 2, 4, 8, 16],
                "decrease_axis": [],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSliceOpDecreaseAxisTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSliceOpDecreaseAxisTest"
        self.cls = TestSliceOp
        self.inputs = [
            {
                "shape": [1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
                "decrease_axis": [0],
            },
            {
                "shape": [1, 1, 1, 1, 1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
                "decrease_axis": [1, 2, 3],
            },
            {
                "shape": [1, 1, 1, 1, 1],
                "axes": [0],
                "starts": [0],
                "ends": [1],
                "strides": [1],
                "decrease_axis": [0, 1, 2, 3, 4],
            },
            {
                "shape": [128, 32],
                "axes": [0, 1],
                "starts": [127, 0],
                "ends": [128, 32],
                "strides": [16, 2],
                "decrease_axis": [0],
            },
            {
                "shape": [32, 10, 64],
                "axes": [0, 2],
                "starts": [31, 32],
                "ends": [32, 33],
                "strides": [2, 4],
                "decrease_axis": [0, 2],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


if __name__ == "__main__":
    TestSliceOpLegacyTestCase().run()
    TestSliceOpShapeTest().run()
    TestSliceOpDtypeTest().run()
    TestSliceOpAxesTest().run()
    TestSliceOpStridesTest().run()
    TestSliceOpDecreaseAxisTest().run()
