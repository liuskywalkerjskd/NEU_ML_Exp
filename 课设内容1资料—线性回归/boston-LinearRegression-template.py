import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # 1.从文件导入数据

    # 2.将原数据集拆分成训练集和测试集，按照test_rate划分

    # 3.返回训练和测试数据集

    return

def data_processing():
    # 传入train_data,test_data参数

    # 1.计算数据max,min,均值，标准差

    # 2.分别对训练和测试数据集进行标准化处理（z-score 标准化）

    # 3.归一化处理（不必须）

    # 4.返回处理后的数据集

    return

class LinearRegression_numpy(object):
    def __init__(self, num_of_weights):
        # 初始化系数w的值（可随机产生random.randn()，也可采用ones()赋值）。
        # 若系数w的维度可设置为=特征数量+1（包含截距），则类成员变量有系数w。
        # 若系数w的维度也可设置为=特征数量，需另外初始化截距b，类成员变量有参数w和b。


    # 将预测输出的过程以“类和对象”的方式来描述。
    def forward(self, ):
        # 通过forward函数（“前向计算”）完成从特征和参数到输出预测值的计算过程
        # 使用train_data数据和系数w(含截距）进行计算

        # 将计算结果返回
        return


    def loss(self, ):
        # 最常采用的衡量方法是使用均方误差（MSE）作为评价模型好坏的指标
        # 使用forward方法计算的结果与实际y数据，求MSE

        # 返回MSE
        return cost

    def gradient(self,):

        # 1、调用forward方法计算残差

        # 2、计算梯度（梯度维度与系数w的维度相同）
        # 若是BGD算法，则总梯度是对每个样本对梯度贡献的平均值

        # 3、返回梯度值
        return

    # 系数w（含截距）更新
    def update(self, ):
        # eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率
        # 根据梯度和学习率更新系数w（包含截距）
        # 更新类属性，无需返回值


    def train_BGD(self, training_data, num_epoches, eta):  # 训练代码

        # 代表样本集合要被训练遍历几次
        for epoch_id in range(num_epoches):
            # 1.经典的四步训练流程：前向计算->计算损失->计算梯度->更新参数（分别调用类的方法）

            # 2.将每次迭代的损失函数的值存起来，用于绘制损失曲线（也可在循环中输出每次迭代的损失值，查看中间过程）

            # 3.可将损失函数数组/列表返回

        return

# 1、获取数据并划分数据集

# 2、数据标准化/归一化处理

# 3、LinearRegression_numpy类实例化
# 训练模型，调用类中的train_BGD方法

# 4、使用测试集进行测试，模型评价
# 将模型训练得到的系数w与测试数据test_data计算得到预测值。
# 计算测试集的MSE

# 5、画出损失函数的变化趋势，画出预测值与真实值曲线
