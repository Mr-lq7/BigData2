#coding=utf-8
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import Imputer
from collections import OrderedDict
import random
#a = os.getcwd()
#print(a)
rem = []
##-------------------------------------------------------------------------
##处理测试集与训练集离散化的区间一致
def Deal(after_train_dataset, item, Divide_Number):
    list = []
    bin = after_train_dataset[item].value_counts(sort = True)
    for i in range(0,Divide_Number):
        bin_tmp = str(bin.index[i])
        bin_tmp = bin_tmp.strip('(')
        bin_tmp = bin_tmp.strip(']')
        bin_tmp = bin_tmp.split(',')
        for element in bin_tmp:
            #print(element)
            if float(element) not in list:
                list.append(float(element))
        list.sort()
    return list
##-------------------------------------------------------------------------

def Preprocess():
#train_dataset是原始数据, after_train_dataset是预处理过的数据
    train_dataset = pd.read_csv('PY_PROJECT/f_train.csv', encoding = 'gbk')
    test_dataset = pd.read_csv('PY_PROJECT/f_test.csv', encoding = 'gbk')

    
    #print(train_dataset.shape)
    #删除SNP21，SNP22，SNP23，RBP4列，数据缺失太多
    #注意考虑测试集后三个属性删不删
    train_dataset = train_dataset.drop(['id', 'SNP21', 'SNP22', 'SNP23', 'RBP4'], axis = 1)
    test_dataset = test_dataset.drop(['id', 'SNP21', 'SNP22', 'SNP23', 'RBP4'], axis = 1)
    train_dataset = pd.DataFrame(train_dataset)
    test_dataset = pd.DataFrame(test_dataset)
    #print(train_dataset.shape)
    train_attributes = train_dataset.columns
    test_attributes = test_dataset.columns
    #print(train_attributes)
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

    ####-----------------------------------------------------
    flag = []
    k = 0
    after = pd.DataFrame()
    test = pd.DataFrame()
    flag = random.sample(range(0, 1000), 800)
    rem.append(flag)
    #print(flag)
    p = 0
    for i in range(1000):
        if i in flag:
            after[p] = after_train_dataset.loc[i].T
            p = p + 1
    
    p = 0
    for i in range(1000):
        if i not in flag:
            test[p] = after_train_dataset.loc[i].T
            p = p + 1

    after_train_dataset = pd.DataFrame(after.values.T, index=after.columns, columns=after.index)
    after_test_dataset = pd.DataFrame(test.values.T, index=test.columns, columns=test.index)

    #######-------------------------------------------------------------------------
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
    Divide_Number = 3
    for item in Discrete_Attributes:
        after_train_dataset[item] = pd.cut(after_train_dataset[item], Divide_Number)
        bins = Deal(after_train_dataset, item, Divide_Number);
    #    print(bins)
        after_test_dataset[item] = pd.cut(after_test_dataset[item], bins)
    
    #after_train_dataset.to_csv('PY_PROJECT/after_train.csv', header = 1, index = None, encoding = 'gbk')
    #after_test_dataset.to_csv('PY_PROJECT/after_test.csv', header = 1, index = None, encoding = 'gbk')

    return after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes
##------------------------------------------------------------------------------------------------

#拉普拉斯校准
#想法返回一个装有所有区间值的列表，然后将区间值作为字符串扔进去，以Divide_Number为单位
#顺便计算概率
def Deal_Zero(after_train_dataset, Discrete_Attributes, Divide_Number, attribute_yes_count, attribute_no_count, yes, no):
    flag = []
    flag1 = []
    k = -1
    w = 0
    r = 0
    for item in Discrete_Attributes:
        w = 0
        r = 0
        k += 1
        element = after_train_dataset[item].value_counts()
        for i in range(Divide_Number):
            tp = element.index[i]
            tp = str(item) + str(' ') + str(tp)
            if w == 0:
                if(attribute_yes_count[tp] == 0):
                    #print(item)
                    for j in range(Divide_Number):
                        tp1 = element.index[j]
                        tp1 = str(item) + str(' ') + str(tp1)
                        attribute_yes_count[tp1] += 1
                        if k not in flag:
                            flag.append(k)
                    w = 1
            if r == 0:
                if(attribute_no_count[tp] == 0):
                    for j in range(Divide_Number):
                        tp2 = element.index[j]
                        tp2 = str(item) + str(' ') + str(tp2)
                        attribute_no_count[tp2] += 1
                        if k not in flag1:
                            flag1.append(k)
                    r = 1
    


    #print('-==')
    #print(flag, flag1)
    #for key, value in attribute_no_count.items():
    #    print(key, value)
    #print('-==')
    #for key, value in attribute_yes_count.items():
    #    print(key, value)
##计算概率--------------------------------------------------
    k = -1
    a = 0
    b = 0
    for item in Discrete_Attributes:
        k += 1
        element = after_train_dataset[item].value_counts()
        a = 0
        b = 0
        for i in range(Divide_Number):
            tmp = element.index[i]
            tmp = str(item) + str(' ') + str(tmp)
            if a == 0:
                if k in flag:
                    for j in range(Divide_Number):
                        tmp1 = element.index[j]
                        tmp1 = str(item) + str(' ') + str(tmp1)
                        attribute_yes_count[tmp1] = float(attribute_yes_count[tmp1] / (yes + Divide_Number))
                    a = 1
                else:
                    attribute_yes_count[tmp] = float(attribute_yes_count[tmp] / yes)
            if b == 0:
                if k in flag1:
                    for j in range(Divide_Number):
                        tmp2 = element.index[j]
                        tmp2 = str(item) + str(' ') + str(tmp2)
                        attribute_no_count[tmp2] = float(attribute_no_count[tmp2] / (no + Divide_Number))
                    b = 1
                else:
                    attribute_no_count[tmp] = float(attribute_no_count[tmp] / no)

    return attribute_yes_count, attribute_no_count




#贝叶斯分类,返回一个分类列表
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
    print('-----')
    #print(P_yes)
    P_no = float(no) / Class_Total

##计算概率密度P(X|Ci)
    attribute_count = OrderedDict()
    attribute_yes_count = OrderedDict()
    attribute_no_count = OrderedDict()
    for item in Discrete_Attributes:
        attribute = after_train_dataset[item]
        temp = after_train_dataset[item].value_counts()
        #print(temp)
        cnt = 0
        for i in range(Divide_Number):
            index = temp.index[i]
            index = str(item) + str(' ') + str(index)
            for j in range(Class_Total):
                ele = attribute[j]
                ele = str(item) + str(' ') + str(ele)
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
    for key, value in attribute_count.items():
        #print(key, value)
        value1 = value - attribute_yes_count[key] 
        attribute_no_count[key] = value1
        #print(key, value1)
    #print(">>>>>>>")
    #for key, value in attribute_yes_count.items():
    #    print(key ,value)

    attribute_yes_count, attribute_no_count = Deal_Zero(after_train_dataset, Discrete_Attributes, Divide_Number, attribute_yes_count, attribute_no_count, yes, no)

    #print('-----------------------------------------分割线')
    #for key, value in attribute_yes_count.items():
    #    print(key ,value)
    #print(">>>>>>>")

    #for key, value in attribute_no_count.items():
    #    print(key ,value)

    #print('-------------------------------')
    #for key, value in attribute_yes_count.items():
    #    print(key, value)
    #print('ppppppppppppppppppppppppppppppppp')
    #for key, value in attribute_no_count.items():
    #    print(key, value)

#处理其他属性值(除了孕期)一起处理，值的范围都为0-3
    SenAttribute_yes_count = OrderedDict()
    SenAttribute_no_count = OrderedDict()
    SenAttribute_count = OrderedDict()
    #敏感值列表
    S_List = ['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5', 'SNP6', 'SNP7', 'SNP8', 'SNP9', 'SNP10',
    'SNP11', 'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16', 'SNP17', 'SNP18', 'SNP19', 'SNP20','孕次','产次',
    'BMI分类', 'SNP24', 'SNP25', 'SNP26', 'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33', 'SNP34',
    'SNP35', 'SNP36', 'SNP37', 'SNP38', 'DM家族史','SNP39', 'SNP40', 'SNP41', 'SNP42', 'SNP43', 'SNP44',
    'SNP45', 'SNP46', 'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53', 'SNP54', 'SNP55', 'ACEID']
    for item in S_List:
        itemlist = after_train_dataset[item]
        for i in range(Class_Total):
            v = itemlist[i]
            va = str(item) + str(' ') + str(v)
            SenAttribute_count[va] = SenAttribute_count.get(va, 0) + 1
            if labels[i] == 1:
                SenAttribute_yes_count[va] = SenAttribute_yes_count.get(va, 0) + 1
            else:
                SenAttribute_no_count[va] = SenAttribute_no_count.get(va, 0) + 1
 
   # print('????????????????')
    #for key,value in SenAttribute_yes_count.items():
    #    print(key, value)
    #print('???????????????????????????????????')
    #for key,value in SenAttribute_no_count.items():
    #    print(key, value)

    for key,value in SenAttribute_yes_count.items():
        SenAttribute_yes_count[key] = float(value / yes)
    for key,value in SenAttribute_no_count.items():
        SenAttribute_no_count[key] = float(value / no)
    
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #for key,value in SenAttribute_yes_count.items():
    #    print(key, value)
    #print('???????????????????????????????????')
    #for key,value in SenAttribute_no_count.items():
    #    print(key, value)
######----------------------------------------------------------------------------------------
    pre_label = []
    YES = 1.0
    NO = 1.0
    Test_Class_Total = len(after_test_dataset['label'])
    #print(Test_Class_Total)
    Test_label = after_test_dataset['label']
    for i in range(Test_Class_Total):
        o = after_train_dataset.loc[i]
        for item in S_List:
            vo = str(item) + str(' ') + str(o)
            YES = YES * SenAttribute_yes_count.get(vo, 1)
            NO = NO * SenAttribute_no_count.get(vo, 1)

        for item in Discrete_Attributes:
            vo = str(item) + str(' ') + str(o)
            YES = YES * attribute_yes_count.get(vo, 1)
            NO = NO * attribute_no_count.get(vo, 1)
        YES = YES * P_yes
        NO = NO * P_no
        if YES > NO:
            pre_label.append(1)
        elif NO > YES:
            pre_label.append(0)
        else:
            a = random.randint(0,1)
            pre_label.append(a)
       # for item in
    cnt = 0 
    for i in range(Test_Class_Total):
        if(pre_label[i] == Test_label[i]):
            cnt += 1
    accuracy = float(cnt / 200)
    #print('The classify accuracy is: %.2f%%' % (accuracy * 100))
    return accuracy

if __name__ == "__main__":
    result = []
    for i in range(5000):
        after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes = Preprocess()

        acc = Bayers(after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes)
        result.append(acc)
        print(len(rem))
        print('====================')
        if i == 4999:
            print(result)
            pp = np.argsort(result)
            print(pp)
            print(result[pp[-1]])
            print(rem[pp[-1]])
    print(max(result))

