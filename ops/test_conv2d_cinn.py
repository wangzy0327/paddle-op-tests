import time
import os
import numpy as np
import paddle
from paddle import nn

cinn_denied_ops = [
    "arg_max",
    "bitwise_and",
    "concat",
    "cumsum",
    "gather",
    "gather_nd",
    "lookup_table_v2",
    "randint",
    "reduce_sum",
    "reduce_max",
    "slice",
    "strided_slice",
    "roll",
    "tile",
    "transpose2",
    "uniform_random",
    "range",
    "arange",
    "fill_constant",
]

paddle.set_flags({
    "FLAGS_prim_all": True,
    "FLAGS_deny_cinn_ops": ";".join(cinn_denied_ops),
})


def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )

def benchmark(net, input, repeat=10, warmup=3):
    # warm up
    for i in range(warmup):
        net(input)
    paddle.device.synchronize()
    # time
    t = []
    for i in range(repeat):
        t1 = time.time()
        net(input)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("\t\t\t--[benchmark] Run for %d times, the average latency is: %f ms" % (repeat, np.mean(t)))

class TestBase:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.input = self.init_input()
        self.net = None
        self.cinn_net = None

    def init_model(self):
        raise NotImplementedError

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 224, 224])

    def get_net(self, use_cinn):
        if use_cinn:
            if self.cinn_net is None:
                if self.net is None:
                    self.net = self.init_model()
                    self.net.eval()
                self.cinn_net = to_cinn_net(self.net)
                self.cinn_net.eval()
                self.net = None
            return self.cinn_net
        else:
            if self.net is None:
                self.net = self.init_model()
                self.net.eval()
            return self.net

    def eval(self, use_cinn):
        net = self.get_net(use_cinn)
        return net(self.input)

    def check_cinn_output(self):
        #  print("--[check_cinn_output] eval nocinn")
        pd_out = self.eval(use_cinn=False)
        #  print("--[check_cinn_output] eval cinn")
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), pd_out.numpy(), atol=1e-3, rtol=1e-3
        )
        print("\t\t\t--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn, **kwargs):
        print("\t\t\t--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.get_net(use_cinn)
        benchmark(net, self.input, **kwargs)

class TestConv2D(TestBase):
    def init_model(self):
        # 定义仅包含 Conv2D 操作的简单网络
        class Conv2DNet(nn.Layer):
            def __init__(self):
                super(Conv2DNet, self).__init__()
                self.conv = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                return self.conv(x)

        return Conv2DNet()

    def init_input(self):
        # 为 Conv2D 准备输入数据
        return paddle.rand([self.batch_size, 3, 224, 224])  # 假设输入图像大小为 224x224，通道数为 3


### DepthwiseConv2D 测试脚本

class TestDepthwiseConv2D(TestBase):
    def init_model(self):
        # 定义仅包含 DepthwiseConv2D 操作的简单网络
        class DepthwiseConv2DNet(nn.Layer):
            def __init__(self):
                super(DepthwiseConv2DNet, self).__init__()
                self.depthwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)

            def forward(self, x):
                return self.depthwise_conv(x)

        return DepthwiseConv2DNet()

    def init_input(self):
        # 为 DepthwiseConv2D 准备输入数据
        return paddle.rand([self.batch_size, 3, 224, 224])  # 假设输入图像大小为 224x224，通道数为 3


if __name__ == "__main__":
    paddle.set_device("mlu:0")
    print("\t\t\tTest Conv2d  ...")
    model_conv2d = TestConv2D(batch_size=1)
    model_conv2d.check_cinn_output()
    model_conv2d.benchmark(use_cinn=True)