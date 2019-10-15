import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer
#a = os.getcwd()
#print(a)

##-------------------------------------------------------------------------
#处理测试集与训练集离散化的区间一致
def Deal(after_train_dataset, item, Divide_Number):
    list = []
    bin = after_train_dataset[item].value_counts(sort = True)
    #print(bin)
    #print('-------------------------')
    for i in range(0,Divide_Number):
        bin_tmp = str(bin.index[i])
        bin_tmp = bin_tmp.strip('(')
        bin_tmp = bin_tmp.strip(']')
        bin_tmp = bin_tmp.split(',')
        #print(bin_tmp)
        #if i == 0:
         #   list = [float(x) for x in bin_tmp]
        #else:
        for element in bin_tmp:
            print(element)
            if float(element) not in list:
                list.append(float(element))
        list.sort()
    return list
##-------------------------------------------------------------------------

def Preprocess():
#train_dataset是原始数据, after_train_dataset是预处理过的数据
    train_dataset = pd.read_csv('PY_PROJECT/f_train.csv', encoding = 'gbk')
    test_dataset = pd.read_csv('PY_PROJECT/f_test.csv', encoding = 'gbk')

    print(train_dataset.shape)
    #删除SNP21，SNP22，SNP23，RBP4列，数据缺失太多
    #注意考虑测试集后三个属性删不删
    train_dataset = train_dataset.drop(['id', 'SNP21', 'SNP22', 'SNP23', 'RBP4'], axis = 1)
    test_dataset = test_dataset.drop(['id', 'SNP21', 'SNP22', 'SNP23', 'RBP4'], axis = 1)
    train_dataset = pd.DataFrame(train_dataset)
    test_dataset = pd.DataFrame(test_dataset)
    print(train_dataset.shape)
    train_attributes = train_dataset.columns
    test_attributes = test_dataset.columns
    print(train_attributes)
    #dt.to_csv('PY_PROJECT/new.csv', header = 1, index = None)

    #填充缺失值
    train_imputer = Imputer(missing_values = 'NaN',strategy = 'most_frequent', axis = 0, copy = True)
    after_train_dataset = train_imputer.fit_transform(train_dataset)
    after_train_dataset = pd.DataFrame(after_train_dataset)
    after_train_dataset.columns = train_attributes

    test_imputer = Imputer(missing_values = 'NaN',strategy = 'most_frequent', axis = 0, copy = True)
    after_test_dataset = test_imputer.fit_transform(test_dataset)
    after_test_dataset = pd.DataFrame(after_test_dataset)
    after_test_dataset.columns = test_attributes

#after_dt.to_csv('PY_PROJECT/new1.csv', header = 1, index = None, encoding = 'gbk')

#离散化年龄
#bin_age = [16, 25, 35, 50]
#after_dt.年龄 = pd.cut(after_dt.年龄, bin_age)

#离散化分组情况

#离散化身高
#bin_height = [135, 160, 168, 178]

#离散化孕前体重
#bin_weight = [35, 48, 55, 68]

#离散化属性值，需注意的是训练集和测试机的划分区间应一致
    Discrete_Attributes = ['年龄', '身高', '孕前体重', '孕前BMI', '收缩压', '舒张压', '分娩时', '糖筛孕周',
            'VAR00007', 'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC', 'ApoA1', 'ApoB', 'Lpa', 'hsCRP']

#进行离散化，并保证训练集和测试集的离散化一致
    Divide_Number = 4
    for item in Discrete_Attributes:
        after_train_dataset[item] = pd.cut(after_train_dataset[item], Divide_Number)
        bins = Deal(after_train_dataset, item, Divide_Number);
    #    print(bins)
        after_test_dataset[item] = pd.cut(after_test_dataset[item], bins)
    
#    after_train_dataset.to_csv('PY_PROJECT/after_train.csv', header = 1, index = None, encoding = 'gbk')
#    after_test_dataset.to_csv('PY_PROJECT/after_test.csv', header = 1, index = None, encoding = 'gbk')
    return after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes
##------------------------------------------------------------------------------------------------

#拉普拉斯校准
#想法返回一个装有所有区间值的列表，然后将区间值作为字符串扔进去，以Divide_Number为单位
#顺便计算概率
def Deal_Zero(after_train_dataset, Discrete_Attributes, Divide_Number, attribute_yes_count, attribute_no_count, yes, no):
    flag = []
    flag1 = []
    k = -1
    for item in Discrete_Attributes:
        k += 1
        element = after_train_dataset[item].value_counts()
        for i in range(Divide_Number):
            tmp = element.index[i]
            if(attribute_yes_count[tmp] == 0):
                print(item)
                for j in range(Divide_Number):
                    tmp = element.index[j]
                    attribute_yes_count[tmp] += 1
                    if k not in flag:
                        flag.append(k)

            if(attribute_no_count[tmp] == 0):
                for j in range(Divide_Number):
                    tmp = element.index[j]
                    attribute_no_count[tmp] += 1
                    if k not in flag1:
                        flag1.append(k)
    print('-==')
    print(flag, flag1)
    for key, value in attribute_no_count.items():
        print(key, value)
    print('-==')
    for key, value in attribute_yes_count.items():
        print(key, value)
##计算概率--------------------------------------------------
    k = -1
    a = 0
    b = 0
    for item in Discrete_Attributes:
        k += 1
        element = after_train_dataset[item].value_counts()
        for i in range(Divide_Number):
            tmp = element.index[i]
            if k in flag:
                for j in range(Divide_Number):
                    tmp = element.index[j]
                    attribute_yes_count[tmp] = float(attribute_yes_count[tmp] / (yes + Divide_Number))
                a = 1
            else:
                attribute_yes_count[tmp] = float(attribute_yes_count[tmp] / yes)
            if k in flag1:
                for j in range(Divide_Number):
                    tmp = element.index[j]
                    attribute_no_count[tmp] = float(attribute_no_count[tmp] / (no + Divide_Number))
                b = 1
            else:
                attribute_no_count[tmp] = float(attribute_no_count[tmp] / no)
            if a == 1 or b == 1:
                a = 0
                b = 0
                break

    return attribute_yes_count, attribute_no_count

#贝叶斯分类
def Bayers(after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes):

##train-------------------------------------------------------------------------------------------
    Label_Count = {}
    Class_Total = len(after_train_dataset['label'])
    labels = after_train_dataset['label']
    for i in range(0,Class_Total):
        label = labels[i]
        Label_Count[label] = Label_Count.get(label, 0) + 1
    for key, value in Label_Count.items():
        if key == 1:
            yes = value
        else:
            no = value
##计算先验概率P(Ci)
    P_yes = float(yes) / Class_Total
    print(P_yes)
    P_no = float(no) / Class_Total

##计算概率密度P(X|Ci)
    attribute_count = {}
    attribute_yes_count = {}
    attribute_no_count = {}
    print('======')
    for item in Discrete_Attributes:
        attribute = after_train_dataset[item]
        temp = after_train_dataset[item].value_counts()
        print(temp)
        cnt = 0
        for i in range(Divide_Number):
            index = temp.index[i]
            for j in range(Class_Total):
                ele = attribute[j]
                if index == ele:
                    attribute_count[ele] = attribute_count.get(ele, 0) + 1
                    if labels[j] == 1:
                        attribute_yes_count[ele] = attribute_yes_count.get(ele, 0) + 1
                    else:
                        attribute_yes_count[ele] = attribute_yes_count.get(ele, 0)
                else:
                    cnt += 1
            if cnt == Class_Total:
                attribute_count[index] = 0
                attribute_yes_count[index] = 0
                attribute_no_count[index] = 0
            cnt = 0

#Divide_Numbers是区间分割的个数
#attribute_yes_count字典存的是对于离散型变量在label=1时的条件概率
#attribute_yes_count字典存的是对于离散型变量在label=0时的条件概率
    print('pppppppppp')
    for key, value in attribute_count.items():
#        print(key, value)
        value1 = value - attribute_yes_count[key] 
        attribute_no_count[key] = value1
        print(key, value1)

    attribute_yes_count, attribute_no_count = Deal_Zero(after_train_dataset, Discrete_Attributes, Divide_Number, attribute_yes_count, attribute_no_count, yes, no)

    print('-------------------------------')
    for key, value in attribute_yes_count.items():
        print(key, value)
    print('ppppppppppppppppppppppppppppppppp')
    for key, value in attribute_no_count.items():
        print(key, value)



#处理其他属性值(除了孕期)一起处理，值的范围都为0-3






#单独处理孕期，值的范围:0-5


















#    for key, value in attribute_count.items():
#        print(key, value)
    #print('????????')

    #for key, value in attribute_yes_count.items():
    #    print(key, value)


#    for i in range(Divide_Number):
#        temp = after_train_dataset[]
#    a = after_train_dataset['年龄'].value_counts()
#    print(a.index[0])


    

















if __name__ == "__main__":
    after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes = Preprocess()
    Bayers(after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes)
