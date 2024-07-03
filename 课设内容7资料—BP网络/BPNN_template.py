
class BPNN(object):
    def __init__(self):     

        # 根据样本特征数、隐藏层数、每层神经元数量，随机初始化权重和偏置
        
        

    def feedforward(self, x):                       #前向传播

        
        
        return                                  # 返回每层的激活值

    def backpropagation(self,):               #反向传播
        
        # 计算输出层神经元梯度
        
        # 计算隐层神经元梯度
        
        
        return                                    #返回每层的权重梯度和偏置梯度

    def update_weight(self,):   #更新参数
        
        
        
        

    def loss(self,): # 代价函数
    
    
        return


    def relu(self, x):   
   
        return   

    def relu_diff(self, x):    #relu函数导数       
       
        return

    def sigmoid(self, x):
        
        return
    
    def sigmoid_diff(self, x):   #sigmoid函数导数
        
        return 

    def fit(self, training_data, num_epoches, learning_rate):   #训练过程
       
        # num_epoches为迭代次数
        for epoch_id in range(num_epoches):
            # 1.经典的四步训练流程：前向计算->计算损失->计算梯度->更新参数（分别调用类的方法）

            # 2.将每次迭代的代价函数的值存起来，用于绘制损失曲线（也可在循环中输出每次迭代的损失值，查看中间过程）

            # 3.可将损失函数数组/列表返回

        return


if __name__ == '__main__':
  
