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

import unittest

import numpy as np
from op_test import OpTest, is_compile_with_device

import paddle
from paddle.cinn.common import Float
# from paddle.cinn.frontend import NetBuilder
from paddle.cinn import frontend
import time
import numpy as np

class TestBroadcastToOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": np.random.random([6]).astype("float32")}
        self.out_shape = [4, 5, 6]
        self.broadcast_axes = [2]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)        
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time()        
        out = paddle.broadcast_to(x, shape=self.out_shape)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)

        print(f"Paddle Execution time: {execution_time:.6f} seconds")
        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("BroadcastTo")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        print("CINN running at ", target.arch)        
        out = builder.broadcast_to(
            x, out_shape=self.out_shape, broadcast_axes=self.broadcast_axes
        )
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["x"]
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
        self.check_outputs_and_grads(all_equal=True)


class TestBroadcastToCase1(TestBroadcastToOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([1, 1, 3]).astype("float32")}
        self.out_shape = [4, 5, 3]
        self.broadcast_axes = [0, 1, 2]


class TestBroadcastToCase2(TestBroadcastToOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([5, 3]).astype("float32")}
        self.out_shape = [4, 5, 3]
        self.broadcast_axes = [1, 2]


# class TestBroadcastToCase3(TestBroadcastToOp):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([4, 3]).astype("float32")}
#         self.out_shape = [4, 5, 3]
#         self.broadcast_axes = [0, 2]

#     def test_check_results(self):
#         self.build_cinn_program(self.target)
#         # because paddle and numpy do not support discontinuous broadcast,
#         # so here we just pass the check until we know how to compose
#         pass


# class TestBroadcastToCase4(TestBroadcastToOp):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([5]).astype("float32")}
#         self.out_shape = [4, 5, 3]
#         self.broadcast_axes = [1]

#     def test_check_results(self):
#         self.build_cinn_program(self.target)
#         # because paddle and numpy do not support discontinuous broadcast,
#         # so here we just pass the check until we know how to compose
#         pass


class TestBroadcastToOpNoAxes(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))        
        self.init_case()

    def init_case(self):
        self.inputs = {"x": np.random.random([6]).astype("float32")}
        self.out_shape = [4, 5, 6]

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)        
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        # 记录开始时间
        start_time = time.time()        
        out = paddle.broadcast_to(x, shape=self.out_shape)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        # print(out)

        print(f"Paddle Execution time: {execution_time:.6f} seconds")
        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = frontend.NetBuilder("BroadcastTo")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        print("CINN running at ", target.arch)        
        out = builder.broadcast_to(x, out_shape=self.out_shape)
        
        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.inputs["x"]
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


# class TestBroadcastToNoAxesCase1(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([1, 1, 3]).astype("float32")}
#         self.out_shape = [4, 5, 3]


# class TestBroadcastToNoAxesCase2(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([5, 3]).astype("float32")}
#         self.out_shape = [4, 5, 3]


# class TestBroadcastToNoAxesCase3(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([4, 1, 3]).astype("float32")}
#         self.out_shape = [4, 5, 3]


# class TestBroadcastToNoAxesCase4(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([1, 1, 1]).astype("float32")}
#         self.out_shape = [4, 5, 3]


# class TestBroadcastToNoAxesCase5(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([5]).astype("float32")}
#         self.out_shape = [4, 5, 3]

#     def test_check_results(self):
#         self.build_cinn_program(self.target)
#         # because paddle and numpy do not support discontinuous broadcast,
#         # so here we just pass the check until we know how to compose
#         pass


# class TestBroadcastToNoAxesCase6(TestBroadcastToOpNoAxes):
#     def init_case(self):
#         self.inputs = {"x": np.random.random([1]).astype("float32")}
#         self.out_shape = [5]


if __name__ == "__main__":
    # TestBroadcastToCase1().run()
    # TestBroadcastToCase2().run()
    unittest.main()
