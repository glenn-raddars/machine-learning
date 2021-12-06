from sklearn.datasets import load_iris
import sklearn.model_selection as sms
from sklearn.tree import DecisionTreeClassifier,export_graphviz
"""鸢尾花数据集查看决策树效果https://blog.csdn.net/mn_kw/article/details/79913786"""

def decision_iris():
    iris = load_iris()

    x_train,x_test,y_train,y_test = sms.train_test_split(iris.data,iris.target,random_state=22)
    """实例化决策树"""
    estimator = DecisionTreeClassifier(criterion="entropy")#criterion参数是选择决策树类型，默认是基尼系数，这里选择entropy是信息增益
    #还有max_depth这个参数决定决策树深度，过高会过拟合
    estimator.fit(x_train, y_train)
    """评估模型"""
    y_predict = estimator.predict(x_test)
    print("预测结果:\n",y_predict)
    print("预测对错:\n",y_predict == y_test)
    print("预测准确率:\n",estimator.score(x_test, y_test))
    #可视化决策树
    export_graphviz(estimator,out_file="C:/python/machine_learning/output/iris_tree.dot",feature_names=iris.feature_names)
    """图中的entropy是分到这个类时的信息增益，samples是这个类里有的样本数量"""

if __name__ == "__main__":
    decision_iris()