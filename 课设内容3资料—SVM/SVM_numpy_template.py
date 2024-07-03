import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


class Supper_Vector_Machine:
    def __init__(self, dataset, target, C, toler, Iter_max):
        """
           参数:
           dataset : 训练样本数据
           target : 训练样本标签
           C ：惩罚系数
           toler ： 终止条件（迭代阈值）
           Iter_max ： 终止条件（最大迭代次数）
           属性：
           alpha ： SVM对偶问题的拉格朗日乘子（乘子个数=样本数据量）
           w(coef_) ：线性模型（决策面）的系数
           b ：线性模型（决策面）的截距
        """


    def Fx(self, i):
        """
           f(x):决策函数
           参数:i (一个待优化的拉格朗日乘子a[i]的下标)
        """
        # 将第i行样本带入决策函数中计算。
        # 返回：计算结果f(xi)


    def Kernel(self, i, j):
        """
           参数:i，j (两个待优化的拉格朗日乘子a[i]和a[j]的下标)

        """
        # Kernel(xi, xj): 核函数计算（线性核就是x_i ^ T * x_j）
        # 返回：计算结果


    def random_j(self, i):
        """
           参数:i(一个待优化的拉格朗日乘子a[i]的下标)
        """
        # 随机选择另一个待优化的拉格朗日乘子a[j]的下标j(j与i不相同）
        # 返回：j


    def get_L_H(self, i, j):
        """
           参数:i，j (两个待优化的拉格朗日乘子a[i]和a[j]的下标)
        """
        # 计算上下界
        # 返回：上界和下界


    def filter(self, L, H, alpha_j):
        """
           参数:
            i，j：两个待优化的拉格朗日乘子a[i]和a[j]的下标
            L, H：上下界
        """
        # 按边界对a[j]值进行修剪，使其在（L, H）范围内。
        # 返回：修剪后的a[j]

    def SMO(self):

        # 外循环：迭代次数
        # change_num用于记录拉格朗日乘子更新的次数，iter用于记录遍历拉格朗日乘子的次数
        iter = 0
        while iter < self.iter_max:
            change_num = 0
            # 内循环：遍历拉格朗日乘子，作为a[i]
            for i in range(样本量):
                # 1、计算Fx(i)，Ei


                # 2、随机选择另一个要优化的拉格朗日乘子a[j]：random_j方法


                # 3、计算Fx(j)，Ej


                # 4、计算上下界：get_L_H方法


                    # 判断如果L == H，则a[i]和a[j]都在边界，不用再进行优化，寻找下一对
                    if L == H:
                        continue
                # 5、计算eta：kernel方法

                    # 判断如果eta <= 0，则不再进行优化，寻找下一对
                    if eta <= 0:
                        continue

                # 6、更新a[j]


                # 7、修剪a[j]


                    # 判断如果a[j]_new-a[j]_old < toler，则不更新a[j]，寻找下一对

                # 8、更新a[i]


                # 9、更新bi和bj


                # 10、更新b


                    change_num += 1
            # change_num为0，则表示遍历过一遍所有的拉格朗日乘子，都没有进行更新
            if change_num == 0:
                iter += 1
            else:
                iter = 0 #一旦拉格朗日乘子有一次更新，就重新遍历，直到完成iter_max次遍历，都没有进行更新，就认为找到最优系数
        # 11、迭代完成，计算决策面w

    def predict(self,dataset, target):
        """
           参数:
           dataset : 样本数据
           target : 样本标签
        """

    def score(self, y_true, y_pred):
        """
           参数:
           y_true : 实际标签
           y_pred : 预测标签
        """
        # 评价指标任选


if __name__ == '__main__':

    # 1、导入数据，划分数据集
    # 注意：选择两个特征进行二分类，二分类标签为1和-1

    # 2、创建SVM对象，训练模型
    svm = Supper_Vector_Machine(x_train, y_train,C, toler, Iter_max)
    svm.SMO()
    # 输出决策面系数w和截距b

    # 3、画出特征散点图和决策面，标出支持向量（通过可视化效果，判断程序编写是否正确）

    # 4、使用测试集带入模型预测

    # 5、模型评价


