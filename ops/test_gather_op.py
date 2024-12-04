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

import logging
import os

import numpy as np
from op_test import OpTest, OpTestTool, is_compile_with_device
from op_test_helper import TestCaseHelper

import paddle
from paddle.cinn.common import Int, is_compiled_with_cuda
# from paddle.cinn.frontend import NetBuilder

from paddle.cinn import frontend
import numpy as np
import time

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="gather")


@OpTestTool.skip_if(
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestGatherOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info)) 
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.data = None

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)         
        inputs = self.case
        dtype = self.case["x_dtype"]
        axis = inputs["axis"]
        x_shape = inputs["x"]
        index_shape = inputs["index"]
        # Paddle does not support negative axis values.
        axis = axis if axis >= 0 else len(x_shape) + axis
        x = np.random.randn(*x_shape).astype(dtype)
        index = np.random.randint(0, x_shape[axis], index_shape).astype("int32")
        self.data = [x, index]
        x = paddle.to_tensor(x, stop_gradient=False)
        index = paddle.to_tensor(index, stop_gradient=False)
        # 记录开始时间
        start_time = time.time()         
        out = paddle.gather(x, index, axis)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)
        
        print(f"Paddle Execution time: {execution_time:.6f} seconds")        
        logger.debug(f" -- The output of Paddle:\n{out}")
        self.paddle_outputs.append(out)

    def build_cinn_program(self, target):
        inputs = self.case
        dtype = self.case["x_dtype"]
        axis = inputs["axis"]
        print("CINN running at ", target.arch)        
        builder = frontend.NetBuilder("gather")
        x = builder.create_input(self.nptype2cinntype(dtype), inputs["x"], "x")
        index = builder.create_input(Int(32), inputs["index"], "index")
        out = builder.gather(x, index, axis=axis)
        
        # computation = frontend.Computation.build_and_compile(target, builder)        
        
        # tensor_data = [
        #     inputs["x"],
        # ]
        
        # computation.get_tensor("x").from_numpy(tensor_data[0], target)
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
        res = self.get_cinn_output(prog, target, [x, index], self.data, [out])
        logger.debug(f" -- The output of CINN:\n{res}")
        res_tensor = paddle.to_tensor(res)
        self.cinn_outputs.extend(res_tensor)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestGatherOpAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestGatherOpAll"
        self.cls = TestGatherOp
        # note: The possible values of axis are related to x, so axis is added in self.inputs
        self.inputs = [
            {"x": [128], "index": [64], "axis": 0},
            # {"x": [16, 32], "index": [32], "axis": 0},
            # {"x": [16, 32], "index": [32], "axis": 1},
            # {"x": [8, 16, 32], "index": [16], "axis": -3},
            # {"x": [8, 16, 32], "index": [8], "axis": -2},
            # {"x": [8, 16, 32], "index": [8], "axis": -1},
            # {"x": [8, 16, 32], "index": [4], "axis": 2},
            # {"x": [16, 8, 4, 64], "index": [4], "axis": 2},
            # {"x": [16, 8, 4, 1024], "index": [4], "axis": 2},
            # {"x": [16, 8, 4, 1], "index": [4], "axis": 2},
            # {"x": [1, 1, 1, 1], "index": [4], "axis": 2},
        ]
        self.dtypes = [
            # {"x_dtype": "int16", "y_dtype": "int64"},
            # {"x_dtype": "int32", "y_dtype": "int64"},
            # {"x_dtype": "int64", "y_dtype": "int64"},
            {"x_dtype": "float32", "y_dtype": "int32"},
            # {"x_dtype": "float64", "y_dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestGatherOpAll().run()
