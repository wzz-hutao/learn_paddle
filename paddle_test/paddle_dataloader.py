# Paddle的数据加载方式：
# 框架已经写好了数据加载格式，我们只需要自定义BasicDataLoader类即可。
import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid


# 数据增强类
class Transform(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, input, label):
        # 对输入的数据和标签进行增强
        input = cv2.resize(input, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return input, label


# 基础数据加载类
class BasicDataLoader(object):
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):  # 打乱
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle

        self.data_list = self.read_list()  # data_list属性的值为read_list()函数的返回值

    def read_list(self):
        data_list = []
        # 读取文件
        with open(self.image_list_file) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder, line.split()[1])
                data_list.append((data_path, label_path))

        random.shuffle(data_list)
        return data_list

    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape

        assert h == h_gt, "ERROR"
        assert w == w_gt, "ERROR"

        if self.transform:
            data, label = self.transform(data, label)

        label = label[:, :, np.newaxis]  # 给lable多一维

        return data, label

    # 方法复写
    def __len__(self):
        return len(self.data_list)

    def __call__(self):
        for data_path, label_path in self.data_list:
            """
            imread(图片路径，读取图片的形式）
            cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
            cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
            """
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # BGR->RGB
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            print(data.shape, label.shape)
            data, label = self.preprocess(data, label)
            yield data, label


def main():
    batch_size = 5
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        transform = Transform(256)
        # TODO: create BasicDataloder instance
        basic_dataloader = BasicDataLoader(
            image_folder=r"C:\Users\wzzyyds\PycharmProjects\pythonProject1\learn_paddle\paddle_test\work\dummy_data",
            image_list_file=r"C:\Users\wzzyyds\PycharmProjects\pythonProject1\learn_paddle\paddle_test\work\dummy_data\list.txt",
            transform=transform,
            shuffle=True
        )

        # TODO: create fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)  # 创建paddle的数据加载器

        # TODO: set sample generator for fluid dataloader  为paddle的数据加载器设置参数
        dataloader.set_sample_generator(basic_dataloader,
                                        batch_size=batch_size,
                                        places=place)

        num_epoch = 2
        for epoch in range(1, num_epoch + 1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')


if __name__ == "__main__":
    main()
