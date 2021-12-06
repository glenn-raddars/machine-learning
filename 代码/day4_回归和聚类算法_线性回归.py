from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
"""线性模型定义:自变量一次 y = w1x1 + w2x2 + ... + b；
参数一次 y = w1x1 + w2x1^2 + w3x1^3 + w4x2 + ... + b
所以，线性关系一定是线性模型，而线性模型不一定是线性关系"""

"""线性回归之正规方程法https://zhuanlan.zhihu.com/p/34842727"""
"""线性回归之梯度下降法https://zhuanlan.zhihu.com/p/90462431"""

def linear1_zheng_gui():
    boston = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    #开始实例化估计器
    estimator = LinearRegression()#fit_intercept=True,默认计算偏置
    estimator.fit(x_train, y_train)
    #输出模型
    print("正规方程的权重系数:\n",estimator.coef_)
    print("偏置为:\n",estimator.intercept_)

    """评价梯度下降和正规方程的性能——均方误差"""
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)
    print("正规方程的均方误差:\n",error)

def linear2_ti_du():
    boston = load_boston()
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    #开始实例化估计器
    estimator = SGDRegressor()#fit_intercept=True,默认计算偏置,它的默认参数一般就很好了
    estimator.fit(x_train, y_train)
    #输出模型
    print("梯度下降的权重系数:\n",estimator.coef_)
    print("偏置为:\n",estimator.intercept_)

    """评价梯度下降和正规方程的性能——均方误差"""
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test,y_predict)
    print("梯度下降的均方误差:\n",error)


if __name__ == "__main__":
    linear1_zheng_gui()
    linear2_ti_du()
    