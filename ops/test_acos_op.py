# Copyright (c) 2023 CINN Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

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
import time
import numpy as np


@OpTestTool.skip_if(
    not is_compile_with_device, "x86 test will be skipped due to timeout."
)
class TestAcosOp(OpTest):
    def setUp(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=-1,
            high=1,
        )

    def build_paddle_program(self, target):
        print("Paddle running at ", target.arch)
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        # 记录开始时间
        start_time = time.time()
        out = paddle.acos(x)
        end_time = time.time()
        # 计算执行时间
        execution_time = end_time - start_time
        print(out)

        print(f"Paddle Execution time: {execution_time:.6f} seconds")

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        print("CINN running at ", target.arch)        
        builder = frontend.NetBuilder("acos")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        
        print("CINN running at ", target.arch)

        out = builder.acos(x)

        computation = frontend.Computation.build_and_compile(target, builder)
        
        tensor_data = [
            self.x_np
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
        print(output)
        self.cinn_outputs = [output]
        # prog = builder.build()
        # 记录开始时间
        # start_time = time.time()
        # res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        # end_time = time.time()
        # # 计算执行时间
        # execution_time = end_time - start_time

        # print(f"CINN Execution time: {execution_time:.6f} seconds")

        # self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestAcosCase1(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestAcosCase1"
        self.cls = TestAcosOp
        self.inputs = [{"x_shape": [512, 256]}]
        self.dtypes = [
            {"x_dtype": "float32"},
            # {
            #     "x_dtype": "float64",
            # },
        ]
        self.attrs = []


class TestAcosCase2(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestAcosCase2"
        self.cls = TestAcosOp
        self.inputs = [
            # {"x_shape": [1]},
            {"x_shape": [1024]},
            # {"x_shape": [512, 256]},
            # {"x_shape": [128, 64, 32]},
            # {"x_shape": [128, 2048, 32]},
            # {"x_shape": [16, 8, 4, 2]},
            # {"x_shape": [1, 1, 1, 1]},
            # {"x_shape": [16, 8, 4, 2, 1]},
        ]
        self.dtypes = [{"x_dtype": "float32"}]
        self.attrs = []


if __name__ == "__main__":
    # TestAcosCase1().run()
    TestAcosCase2().run()
