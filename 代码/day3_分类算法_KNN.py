from sklearn.datasets import load_iris
import sklearn.model_selection as sms 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
"""Knn算法，又称K邻近算法，分类方法就是先拿一组数据惊醒训练分类，然后将要分类的数据与这个训练好的数据进行比对，
哪些样本离他最近，他就属于那些样本所属的类别"""

def knn_demo():
    iris = load_iris()
    #划分数据集
    x_train,x_test,y_train,y_test = sms.train_test_split(iris.data,iris.target,random_state=6)

    #标准化测试机，训练集
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)#为什么不是fit_transform，因为fit是专门做计算的，然后这里训练集和测试集要统一，就都用训练集的方差和均值

    #开始分类
    estimator = KNeighborsClassifier(n_neighbors=3)#主要看几个邻居属于什么类别，这里是看三个
    
    estimator.fit(x_train, y_train)#训练模型
    #比对预测值和真实值
    y_predict = estimator.predict(x_test)
    print("预测的结果为：\n",y_predict)
    print("预测值与真实值比对:\n",y_test == y_predict)
    #用score函数计算准确度
    score = estimator.score(x_test, y_test)
    print("计算准确度:\n",score)


"""应用网格搜索和交叉验证的KNN算法"""
def knn_demo_gridsearchCV():
    """网格搜索：给一组估计其的参数，一个一个试，试出最佳参数"""
    """交叉验证：k折交叉验证将所有数据集分成k份，不重复地每次取其中一份做测试集，
    用其余k-1份做训练集训练模型，之后计算该模型在测试集上的得分,将k次的得分取平均得到最后的得分,
    最后取得分最高的分发训练估计器,这个交叉验证是用来辅助找出最佳的参数"""

    iris = load_iris()
    #划分数据集
    x_train,x_test,y_train,y_test = sms.train_test_split(iris.data,iris.target,random_state=6)

    #标准化测试机，训练集
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)#为什么不是fit_transform，因为fit是专门做计算的，然后这里训练集和测试集要统一，就都用训练集的方差和均值

    #开始分类
    estimator = KNeighborsClassifier()#主要看几个邻居属于什么类别，这里是看三个
    param_grid = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = sms.GridSearchCV(estimator,param_grid=param_grid,cv=10)#这就是在训练，param_grid是网格训练参数，cv是交叉验证参数
    estimator.fit(x_train, y_train)#训练模型
    #比对预测值和真实值
    y_predict = estimator.predict(x_test)
    print("预测的结果为：\n",y_predict)
    print("预测值与真实值比对:\n",y_test == y_predict)

    #用score函数计算准确度
    score = estimator.score(x_test, y_test)
    print("计算准确度:\n",score)
    #最佳参数
    print("最佳参数:\n",estimator.best_params_)
    #最佳结果
    print("最佳结果:\n",estimator.best_score_)
    #最佳估计器
    print("最佳估计器:\n",estimator.best_estimator_)
    #交叉验证结果
    print("交叉验证结果:\n",estimator.cv_results_)

if __name__ == "__main__":
    #knn_demo()
    knn_demo_gridsearchCV()