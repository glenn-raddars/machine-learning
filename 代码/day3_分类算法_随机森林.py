from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
"""随机森林算法讲解https://blog.csdn.net/yangyin007/article/details/82385967"""
"""讲RandomForestClassifier的API参数的https://blog.csdn.net/w952470866/article/details/78987265/"""

"""加入网格搜索的随机森林"""

def random_forest():
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22)
    #实例化随机森林，参数都由网格搜索和交叉验证找出最佳
    estimator = RandomForestClassifier()
    #开始分类
    param_grid = {"n_estimators":[3,5,7,9,11],"max_depth":[5,6,7,8,9]}
    estimator = GridSearchCV(estimator,param_grid=param_grid,cv=3)#这就是在训练，param_grid是网格训练参数，cv是交叉验证参数
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
    random_forest()