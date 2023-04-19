import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable  # 转换为paddle数据格式
from paddle.fluid.dygraph import Pool2D  #TODO
from paddle.fluid.dygraph import Conv2D  #TODO
import numpy as np
np.set_printoptions(precision=2)


class BasicModel(fluid.dygraph.Layer):
    # BasicModel contains:  BasicModel就是我们的网络
    # 1. pool:   4x4 max pool op, with stride 4
    # 2. conv:   3x3 kernel size, takes RGB image as input and output num_classes channels,
    #            note that the feature map size should be the same
    # 3. upsample: upsample to input size
    #
    # TODOs:
    # 1. The model takes an random input tensor with shape (1, 3, 8, 8)
    # 2. The model outputs a tensor with same HxW size of the input, but C = num_classes
    # 3. Print out the model output in numpy format

    def __init__(self, num_classes=59):
        super(BasicModel, self).__init__()
        # 属性即为paddle中的操作函数
        self.pool = Pool2D(pool_size=2,pool_stride=2)  # 定义池化操作
        self.conv = Conv2D(num_channels=3, num_filters=num_classes,filter_size=1)  # 定义卷积操作

    def forward(self, inputs):  # inputs的维度（n,c,h,w）
        x = self.pool(inputs)  # 最大池化，图片大小缩小一半
        x = fluid.layers.image_resize(x, out_shape=(inputs.shape[2], inputs.shape[3]))  # 上采样,维度为out_shape(h,w)
        x = self.conv(x)  # 卷积操作，提取特征
        return x

def main():
    place = paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = BasicModel(num_classes=59)
        model.eval()
        input_data = np.random.rand(1,3,8,8).astype(np.float32)  # 随机初始化一个np.array量
        print('Input data shape: ', input_data.shape)
        input_data =  to_variable(input_data)  # 将np.array转换为paddle支持的格式
        output_data = model(input_data)  # 通过模型计算输出值
        output_data = output_data.numpy()  # 将模型计算出的tensor转换为numpy
        print('Output data shape: ', output_data.shape)

if __name__ == "__main__":
    main()
