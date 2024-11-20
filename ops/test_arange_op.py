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
class TestArangeOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))        
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "start": self.case["start"],
            "end": self.case["end"],
            "step": self.case["step"],
            "dtype": self.case["dtype"],
        }

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)        
        # 记录开始时间
        start_time = time.time()
        out = paddle.arange(
            self.inputs["start"],
            self.inputs["end"],
            self.inputs["step"],
            self.inputs["dtype"],
        )
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time        
        self.paddle_outputs = [out]
        print(f"Paddle Execution time: {execution_time:.6f} seconds")        

    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("arange")    
        out = builder.arange(
            self.inputs["start"],
            self.inputs["end"],
            self.inputs["step"],
            self.inputs["dtype"],
        )
        print("CINN running at ", target.arch)    

        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["start"],
            self.inputs["end"],
            self.inputs["step"],
            self.inputs["dtype"],
        ]
        
        computation.get_tensor("start").from_numpy(tensor_data[0], target)
        computation.get_tensor("end").from_numpy(tensor_data[1], target)
        computation.get_tensor("step").from_numpy(tensor_data[2], target)
        computation.get_tensor("dtype").from_numpy(tensor_data[3], target)
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
        # res = self.get_cinn_output(prog, target, [], [], [out])

        # self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestArangeOpShapeAndAttr(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestArangeOpShapeAndAttr"
        self.cls = TestArangeOp
        self.inputs = [
            # basic shape test
            # {
            #     "start": 0,
            #     "end": 10,
            #     "step": 1,
            # },
            {
                "start": 0,
                "end": 1024,
                "step": 16,
            },
            # {
            #     "start": 512,
            #     "end": 2600,
            #     "step": 512,
            # },
            # {
            #     "start": 0,
            #     "end": 65536,
            #     "step": 1024,
            # },
            # {
            #     "start": 0,
            #     "end": 131072,
            #     "step": 2048,
            # },
            # {
            #     "start": 0,
            #     "end": 1,
            #     "step": 2,
            # },
            # {
            #     "start": 0,
            #     "end": 1,
            #     "step": 2,
            # },
            # # step test
            # {
            #     "start": 1024,
            #     "end": 512,
            #     "step": -2,
            # },
            # {
            #     "start": 2048,
            #     "end": 0,
            #     "step": -64,
            # },
            # # range test
            # {
            #     "start": -2048,
            #     "end": 2048,
            #     "step": 32,
            # },
            # {
            #     "start": -2048,
            #     "end": -512,
            #     "step": 64,
            # },
            # {
            #     "start": 1024,
            #     "end": 4096,
            #     "step": 512,
            # },
            # {
            #     "start": 1024,
            #     "end": -1024,
            #     "step": -128,
            # },
            # {
            #     "start": -1024,
            #     "end": -2048,
            #     "step": -64,
            # },
            # {
            #     "start": 2048,
            #     "end": 512,
            #     "step": -32,
            # },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestArangeOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestArangeOpDtype"
        self.cls = TestArangeOp
        self.inputs = [
            {
                "start": 5,
                "end": 10,
                "step": 1,
            },
            {
                "start": -10,
                "end": -100,
                "step": -10,
            },
            {
                "start": -10,
                "end": 10,
                "step": 1,
            },
        ]
        self.dtypes = [
            # {"dtype": "int32"},
            # {"dtype": "int64"},
            {"dtype": "float32"},
            # {"dtype": "float64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestArangeOpShapeAndAttr().run()
    TestArangeOpDtype().run()
