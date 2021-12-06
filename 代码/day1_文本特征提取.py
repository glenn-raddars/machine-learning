from sklearn.feature_extraction.text import CountVectorizer
import jieba

def transfer_english_demo():
    text = ["I am Chinese !","You are English dog !","fuck fuck fuck"]
    #实例化转换器
    transfer = CountVectorizer()#统计每个样本特征值出现的次数，还有一个参数是停用词，stop_words=[]，可以把你认为不需要统计的词放进来
    #转换文本
    text_new = transfer.fit_transform(text)
    print("文本为:\n",text_new)
    #转换为正常矩阵，而不是稀疏矩阵，注意，CountVectorizer不提供sparse这个参数
    #所以得用to_arry方法
    print("文本为:\n",text_new.toarray())
    #特征名字
    print("特征名字:\n",transfer.get_feature_names())

def transfer_chinese_demo():
    text = ["你 在 干 甚 么","你 是 个 傻逼","我 今天 很 不 开心"]#必须超过一个字才算特征值
    #实例化转换器
    transfer = CountVectorizer()#统计每个样本特征值出现的次数
    #转换文本
    text_new = transfer.fit_transform(text)
    print("文本为:\n",text_new)
    #转换为正常矩阵，而不是稀疏矩阵，注意，CountVectorizer不提供sparse这个参数
    #所以得用to_arry方法
    print("文本为:\n",text_new.toarray())
    #特征名字
    print("特征名字:\n",transfer.get_feature_names())

def cut_word(text):
    """用于分词"""
    return " ".join(list(jieba.cut(text)))#jion()方法会将字符或字符串用指定的字符连接成一个新的字符串
    #jieba.cut可以讲中文拆分，如把我爱北京天安门拆成 我 爱 北京 天安门

def transfer_chinese_demo2():
    data = ["我爱北京天安门","天安门上太阳升","伟大领袖毛主席","指引我们向前进"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    
    transfer = CountVectorizer()#统计每个样本特征值出现的次数
    #转换文本
    text_new = transfer.fit_transform(data_new)
    print("文本为:\n",text_new)
    #转换为正常矩阵，而不是稀疏矩阵，注意，CountVectorizer不提供sparse这个参数
    #所以得用to_arry方法
    print("文本为:\n",text_new.toarray())
    #特征名字
    print("特征名字:\n",transfer.get_feature_names())



if __name__ == "__main__":
    transfer_english_demo()
    transfer_chinese_demo()
    transfer_chinese_demo2()