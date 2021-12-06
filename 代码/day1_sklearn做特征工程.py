from sklearn.feature_extraction import DictVectorizer
#机器学习只能处理数据，字符串怎么办
#转化成数据，将特征值转化成机器可以读懂的东西——特征工程
#字典特征提取，文本特征提取，图像特征提取

def transfer_dict():
    data = [{"姓名":"包润蛟","性格":'可爱','身高':170},{"姓名":'易淮笠',"性格":"傻逼",'身高':173}]
    #实例化一个转换器
    transfer = DictVectorizer()#默认返回稀疏矩阵
    #转换上述字典
    data_new = transfer.fit_transform(data)
    #他返回的是一个矩阵，使用hotone编码，就是你有几类，就安排几列，你是哪一列，对应为1，其他为零
    #返回的其实是一个稀疏矩阵，不为0的地方才会显示，前面是位置，后面是不为零的数字
    print("data_new:\n",data_new)

    #非稀疏矩阵
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    #最后拿回每一列所表示的含义
    print("data_new\n",transfer.get_feature_names())




if __name__ == "__main__":
    transfer_dict()