from sklearn.datasets import fetch_20newsgroups
import sklearn.model_selection as sms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

"""算法原理
https://zhuanlan.zhihu.com/p/26262151
https://zhuanlan.zhihu.com/p/26329951?utm_medium=social""" 

"""新闻文本分类,使用拉普拉斯平滑的朴素贝叶斯"""

def new_nb_demo():
    np.set_printoptions(threshold=100000000) #全部输出 
    file = open("C:/python/machine_learning/output/朴素贝叶斯分类输出.txt","w")
    news = fetch_20newsgroups(data_home="C:/python/machine_learning/材料",subset="all")#分别是数据集的下载地址和下载内容,all是训练集数据集全要
    #print("新闻数据集为:\n",news)
    #数据集划分
    x_train,x_test,y_train,y_test = sms.train_test_split(news.data,news.target,random_state=7)
    #用tfidf提取特征值
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    #朴素贝叶斯预估器
    estimator = MultinomialNB()#默认alpha为1.0，拉普拉斯平滑的系数为1.0
    estimator.fit(x_train, y_train)
    #模型评估
    y_predict = estimator.predict(x_test)
    print("预测结果:\n",y_predict,file=file)
    print("预测准确与否:\n",y_predict == y_test,file=file)
    print("预测准确度:\n",estimator.score(x_test, y_test),file=file)
    

if __name__ == "__main__":
    new_nb_demo()