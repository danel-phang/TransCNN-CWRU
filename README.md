# TransCNN-CWRU: 精度100%的基于CNN和Transformer的轴承故障分类模型

## 项目简介
本项目实现了一个结合卷积神经网络（CNN）和Transformer的深度学习模型，用于对CWRU轴承故障数据集进行分类。验证与测试精度达到100%

## 数据集
- **数据来源**: CWRU轴承故障数据集。
- **数据格式**: CSV文件，每列为不同类别的振动信号。
- **预处理**:
  - 每个信号被分割为长度为1024的样本块。
  - 每个样本块被标注为对应的故障类别。

## 模型架构
该模型结合卷积神经网络（CNN）和Transformer架构，首先通过两层一维卷积层提取信号的局部特征，并通过最大池化层进行下采样；随后利用Transformer模块中的多头自注意力机制和前馈网络捕获全局特征；最后通过全局平均池化和全连接层完成分类，输出10个故障类别的概率分布。


## 结果与可视化
### 交叉验证结果
- 平均验证精度：`99.8%`。
- 标准偏差：`±0.23%`。
- 训练和验证的准确率与损失曲线可视化。

### 测试集结果
- 测试集精度：`100%`。


## 安装依赖
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
   ```

## 参考
- 凯斯西储大学（CWRU）轴承故障数据集: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)