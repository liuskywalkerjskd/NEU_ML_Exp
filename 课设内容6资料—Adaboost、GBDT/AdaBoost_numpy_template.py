

class DecisionTreeClassifier:
    def __init__(self):
        # 最小的加权分类错误率
        # 最优特征id
        # 最优特征的最优阈值
        # 最优的阈值符号

    def fit(self, X, y, sample_weight):

        # sample_weight为样本权重

        # 遍历特征
            # 遍历阈值
                # 遍历判别符号

                    #计算加权分类误差率(错分样本的权重之和)

                    #记录最小加权分类误差率，最优特征、最优阈值、最优判别符号

        # 将最小加权分类误差率，最优特征、最优阈值、最优判别符号返回
        return

    def predict(self, X):

        # 根据最优特征、最优阈值、最优判别符号，对样本X进行预测

        # 返回预测值1 or -1

        return


class AdaBoostClassifier:
    def __init__(self, n_estimators=50):

        # 基评估器个数

        # 基评估器模型（列表）

        # 基评估器权重（列表）

        # 样本权重（列表）


    def fit(self, X, y):
        # 初始化样本权重

        # 循环创建n_estimators个基评估器

            # 实例化DecisionTreeClassifier，并训练基评估器，加入基评估器模型列表

            # 计算基评估器权重alpha,并加入基评估器权重列表

            # 更新样本权重D,并加入样本权重列表


        return

    def predict(self, X):

        # 遍历训练好的n_estimators个基评估器

            # 把样本放进每个基评估器进行预测（用于后续得到综合加权评分）


        # 根据每个基评估器结果和基评估器权重，获得综合预测结果

        # 返回预测值
        return

    def score(self, X, y):

        # 根据样本X，预测标签y_pred

        # 根据预测标签y_pred和实际标签y，计算分类准确率，并返回结果

        return


    def staged_score(self, X, y):

        # 返回每个基评估器的错误分类概率

        return

if __name__ == "__main__":

    # 1、导入数据集
    # 2、数据集划分
    # 3、实例化
    ABC_clf = AdaBoostClassifier()
    # 4、训练模型（调用fit)
    ABC_clf.fit(x_train, y_train)
    # 5、预测，评分
    ABC_clf.score(X_test, y_test)
