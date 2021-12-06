from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
"""降维就是删去冗余的特征值，有的时候特征值的类过多且繁杂，不需要，这个时候就要选择出无关的特征值，使得机器学习的效率更高"""

"""低方差过滤，将特征值列中，方差过低的列删去"""
def variance_demo():
    iris = load_iris()
    data = iris.data
    #实例化转换器
    transfer = VarianceThreshold(threshold=0.3)#只有三列，有一列的特征值方差小于0.3
    data_new = transfer.fit_transform(data)
    print("低方差过滤后的特征值:\n",data_new,data_new.shape)

"""相关系数法，几个特征值之间的相关系数（公式自己去查，皮尔森相关系数就是上课学的），"""

"""PCA特征降维，用到了SVD奇异式分解"""
def PCA_demo():
    iris = load_iris()
    data = iris.data
    #实例化转换器
    transfer = PCA(n_components=0.95)#对于n_components这个参数，如果传入整数，就是降成几维，如果传入小数，就是保留原有特征的百分之几
    data_new = transfer.fit_transform(data)
    print("低方差过滤后的特征值:\n",data_new,data_new.shape)

if __name__ == "__main__":
    variance_demo()
    PCA_demo()
