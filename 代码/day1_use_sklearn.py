from sklearn.datasets import load_iris #从sklearn.datasets中导入鸢尾花数据集
import sklearn.model_selection as sms
def datasets_iris_use():
    #练习数据集的使用
    iris = load_iris()
    #有两种方式读取数据集 load_* 和 fetch_*,前者是小规模的数据集，后者是大规模的
    #返回类型是一个BUNCH类，继承自字典，返回的都是键值对
    #data，是特征值数组，二维
    #target，是标签数组，一维
    #DESCR,数据描述
    #feature_names，特征名
    #target_names,标签名

    print("鸢尾花数据集：\n",iris)
    print("鸢尾花数据集描述:\n",iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)
    print("查看特征值:\n",iris.data,iris.data.shape)
    print("查看标签数组:\n",iris.target)
    print("查看目标值名字:\n",iris.target_names)

    #数据集的划分，一部分训练，一部分测试
    x_train,x_test,y_train,y_test = sms.train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    #几个参数，按顺序依次是，数据集的特征值，数据集的标签值，测试集的大小（所占百分比），随机数种子(划分是随机采样的)
    #返回依次是 训练及特征值，测试集特征值，训练及目标值，测试集目标值
    print("训练集的特征值:\n",x_train,x_train.shape)
    print("训练及目标值:\n",y_train)

if __name__ == "__main__":
    datasets_iris_use()
