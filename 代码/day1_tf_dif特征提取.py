from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
#下面讲解tfidf提取特征的原理
"""tf是词频，就是这个词在文章中出现的次数，而idf呢
假设有一组文章（文件集）有1000篇文章，“非常”出现在100篇里，而“经济”出现在10篇里，那么“非常”的idf为 lg 1000/100 = 1
 , 而“经济”的idf为 lg 1000/10 = 2，然后tf_idf的判断结果为频率（tf）乘以 idf，这个值越大，证明特征值更有效"""

def cut_word(text):#还是要分词的
    return " ".join(list(jieba.cut(text)))


def  transfer_tf_idf_demo():
    text = ["我爱北京天安门","天安门上太阳升","伟大领袖毛主席","指引我们向前进"]
    #还是要分个词的
    text_new = []
    for sent in text:
        text_new.append(cut_word(sent))

    transfer = TfidfVectorizer()#实例化特征提取器
    print(text_new)
    text_new = transfer.fit_transform(text_new)
    print("稀疏矩阵为:\n",text_new)

    print("真实矩阵为:\n",text_new.toarray())
    print("提取出的特征名称为:\n",transfer.get_feature_names())

if __name__ == "__main__":
    transfer_tf_idf_demo()
