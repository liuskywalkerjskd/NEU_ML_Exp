import warnings
import numpy as np
import pandas as pd



class CART_decision_tree(object):
    """
    一个递归定义的数据结构，用于存储树。
    每个节点可以包含其他节点作为其子节点。
    """

    def __init__(self, tree='cls', criterion='gini', max_depth=None):

        self.feature = None #特征对应列号(索引)
        self.featurename = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0

        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = tree

    def fit(self, features, target):
        """
        该函数接受一个训练数据集，从中建立树状结构来做决定，或者做子节点来做进一步的查询。

        参数:
        features (dataframe): [n * p] n个p属性的观察数据样本
        target (series): [n] 目标值
        """
        self.root = CART_decision_tree()
        if (self.tree == 'cls'):
            self.root._grow_tree(features, target, self.criterion)
        else:
            self.root._grow_tree(features, target, 'mse')
        self.root._prune(self.max_depth)

    def predict(self, features):
        """
        该函数接受一个测试数据集，从中遍历CART树，输出一个 预测数组。

        参数:
        features (dataframe): [n * p] n个p属性的观察数据样本

        返回值:
        list: [n] 预测值
        """
        # 遍历每个测试实例，并反复调用内部方法"_predicts "进行预测
        # 如果当前节点属性低于或高于阈值，直到可以进行预测。
        self.root._predict(每个测试样本)

        return 预测标签

    def print_tree(self):
        """
        用于打印决策树结构的辅助函数。
        """
        self.root._show_tree(0, '')

    def _grow_tree(self, features, target, criterion='gini'):
        """"
        内部函数通过将数据分割成两个分支来增长树的长度，基于一些阈值的值。目前的节点被分配到提供最佳分割的特征，并有相应的阈值。

        参数:
        features (dataframe): [n * p] n个p属性的观察数据样本
        target (series): [n] 目标值
        criterion (string): 分类标准。 default = 'gini'
        """
        self.n_samples = features.shape[0]

        # 如果所有的类都是相同的，节点被标记为该类并返回。终止检索。（待补充）





        if criterion in {'gini', 'entropy'}:
            # 计算哪个类在数据中拥有最多的实例，并为节点分配该标签。（待补充）


        else:
            # 计算回归问题中的平均值，并为节点分配该标签。（待补充）


        # 计算父节点的不纯度。（待补充）


        best_gain = 0.0
        best_feature = None
        best_threshold = None
        # 迭代所有分类决策，以确定最佳信息增益、特征和阈值。
        for col in range(features.shape[1]):

            # 按连续特征：单特征阈值法
            # 创建一个候选阈值的列表thresholds。(待补充）


            # 遍历阈值
            for threshold in thresholds:
                # 计算左侧子节点的数据集标签。（待补充）

                # 计算左侧子节点的不纯度标准。（调用self._calc_impurity方法）（待补充）

                # 计算左侧子节点样本量/父节点样本量的比率。（待补充）


                # 计算右侧子节点的数据集。（待补充）

                # 计算右侧子节点的不纯度标准。（待补充）


                # 计算右侧子节点样本量/父节点样本量的比率。（待补充）


                # 计算信息增益。（待补充）


                if 信息增益 > best_gain:
                    best_gain = 信息增益
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.featurename = 根据特征列号，得到特征名
        self.gain = best_gain
        self.threshold = best_threshold
        # 迭代创建子节点
        self._split_tree(features, target, criterion)

    def _split_tree(self, features, target, criterion):
            """"
            内部函数分割数据集，创建左、右子节点，并逐渐生长树。

            参数:
            features (dataframe): [n * p] n个p属性的观察数据样本
            target (series): [n] 目标值
            criterion (string): 分类标准。 default = 'gini'
            """
            # 划分左侧子节点样本集合（左侧节点特征集合，左侧节点标签集合）

            # 递归树结构
            self.left = CART_decision_tree()
            self.left.depth = self.depth + 1
            self.left._grow_tree(左侧节点特征集合, 左侧节点标签集合, criterion)

            # 划分右侧子节点样本集合（右侧节点特征集合，右侧节点标签集合）


            self.right = CART_decision_tree()
            self.right.depth = self.depth + 1
            self.right._grow_tree(右侧节点特征集合, 右侧节点标签集合, criterion)

    def _calc_impurity(self, criterion, target):
            """"
            内部函数根据选择的措施计算杂质标准并返回数值。

            参数:
            criterion (string): 分类标准
            target (series): [n] 目标值

            返回值:
            float: 不纯度标准的值
            """
            if criterion == 'gini':

                return 计算结果
            elif criterion == 'mse':

                return 计算结果
            else:
                # 计算信息增益

                return 计算结果

    def _prune(self, max_depth):
            """"
            内部函数根据max_depth来调整树的深度。减少了对数据的过度拟合。

            参数:
            max_depth (int): 树的最大深度
            """
            if self.feature is None:
                return

            self.left._prune(max_depth)
            self.right._prune(max_depth)

            pruning = False

            if self.depth >= max_depth:
                pruning = True

            if pruning is True:
                self.left = None
                self.right = None
                self.feature = None

    def _predict(self, d):
            """"
            内部函数接收一行输入值并递归地检查一个阈值，直到返回一个 预测的标签被返回。

            参数:
            d (series): [1 * p] 观察到的p属性的数据样本

            返回值:
            string: 预测的标签
            """
            if self.feature != None:
                if d[self.feature] <= self.threshold:
                    return self.left._predict(d)
                else:
                    return self.right._predict(d)
            else:
                return self.label

    def _show_tree(self, depth, cond):
            """
            内部辅助函数，用于打印树形决策结构。
            """
            base = '|---' * depth + cond
            if self.feature != None:
                print(base + 'if ' + self.featurename + ' <= ' + f"{self.threshold:.2f}")
                self.left._show_tree(depth + 1, 'then ')
                self.right._show_tree(depth + 1, 'else ')
            else:
                print(base + '{class is: ' + str(self.label) + ', number of samples: ' + str(self.n_samples) + '}')

    def score(self, y_true, y_pred):
        """

        :param y_true: 真实值
        :param y_pred: 预测值
        :return: 分类指标/回归指标
        """

if __name__ == "__main__":
    # boston房价回归树
    # 1、导入数据集
    # 2、数据集划分
    # 3、创建回归树实例
    regressionTree = CART_decision_tree(tree='mse', criterion='mse')
    # 4、训练模型（调用fit)
    regressionTree.fit(x_train, y_train)
    # 5、输出树模型
    regressionTree.print_tree()
    # 6、预测，评分


    # iris分类树
    # 1、导入数据集
    # 2、数据集划分
    # 3、创建回归树实例
    classificationTree = CART_decision_tree(tree='cls', criterion='gini')
    # 4、训练模型（调用fit)
    classificationTree.fit(x_train, y_train)
    # 5、输出树模型
    classificationTree.print_tree()
    # 6、预测，评分




