"""
语义分割：给每种object分类
实例分割：给每一种object分类
全景分割：语义 + 示例

语义分割算法的根本目的：像素级分类
语义分割算法的基本流程：
- 输入： 图像RGB
- 算法： 深度学习模型
- 输出： 分析结果（与输入大小一致的单通道图）
- 训练过程：
    - 输入： image + label
    - 前向： out = model(img)
    - 计算损失： loss = loss_func(out,label)
    - 反向： loss.backward()
    - 更新权重： optimizer.minimize(loss)


准确率检测 => 评价指标：
1.mIoU  自己做的和预测的图像像素重叠的部分 ==> （交集）/（并集）  多个类别取mean
2.mAcc->accurary算法 图像像素比较，正确的部分加1，除以整体n
"""