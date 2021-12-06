from sklearn.datasets import load_iris#导入鸢尾花数据集
from sklearn.preprocessing import MinMaxScaler,StandardScaler#导入归一化,标准化API
#归一化原理
#做归一化，是对列进行归一化，原理为两个公式，x' = (x - min)/(max - min)  x'' = x'(mx - mi) + mi,最后取结果x''
# min,max分别为那一列的最大值和最小值,x为要转化的数，mx,mi为要转化的区间[mi,mx]

def minmax_demo():
    """归一化"""
    iris = load_iris()
    data = iris.data
    
    #实例化转换器
    transfer = MinMaxScaler()#feature_range默认(0,1)可以改
    data_new = transfer.fit_transform(data)#开始归一化，主要是方便处理数据，让每个特征类头被学习到
    print("归一化后的数据集特征值:\n",data_new)

"""标准化：
目的也是为了是数据尽可能差异不大，使所有特征都能学习到，也是对列进行处理，公式为 x - mean / σ
mean是均值，σ是标准差"""
def stand_demo():
    """标准化到均值为零，标准差为1,不易受到异常值影响"""
    iris = load_iris()
    data = iris.data
    
    #实例化转换器
    transfer = StandardScaler()#feature_range默认(0,1)可以改
    data_new = transfer.fit_transform(data)#开始归一化，主要是方便处理数据，让每个特征类头被学习到
    print("归一化后的数据集特征值:\n",data_new)

if __name__ == "__main__":
    minmax_demo()
    stand_demo()