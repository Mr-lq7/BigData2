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
    #flag = []
    k = 0
#     after = pd.DataFrame()
#     test = pd.DataFrame()
#     #print(after_train_dataset)
#     #flag = random.sample(range(0, 1000), 800)
#     #rem.append(flag)
#     flag = [670, 314, 31, 319, 321, 412, 309, 222, 830, 281, 921, 631, 382, 354, 458, 121, 20, 526, 266, 220, 282, 702, 740, 704, 937, 676, 818, 761, 785, 877, 162, 492, 297, 168, 936, 495, 13, 923, 963, 337, 944, 433, 488, 871, 909, 210, 964, 563, 377, 101, 735, 623, 775, 700, 515, 553, 394, 577, 276, 265, 489, 397, 523, 672, 366, 582, 68, 972, 749, 299, 385, 527, 900, 674, 470, 530, 485, 200, 238, 261, 580, 864, 901, 25, 330, 992, 824, 767, 56, 331, 634, 746, 696, 361, 924, 718, 482, 796, 544, 137, 115, 665, 845, 269, 469, 617, 229, 317, 647, 567, 983, 243, 540, 782, 536, 182, 637, 956, 848, 650, 587, 828, 503, 158, 431, 244, 464, 517, 198, 726, 344, 584, 134, 578, 755, 295, 422, 446, 96, 902, 461, 479, 910, 335, 486, 907, 728, 343, 535, 434, 396, 91, 183, 922, 407, 62, 5, 645, 640, 769, 618, 538, 620, 322, 428, 318, 973, 894, 838, 450, 110, 562, 368, 252, 294, 102, 713, 873, 609, 663, 727, 502, 176, 777, 405, 822, 413, 435, 737, 781, 675, 76, 454, 872, 336, 778, 371, 996, 417, 163, 174, 677, 751, 112, 410, 542, 552, 378, 61, 180, 858, 189, 932, 583, 927, 149, 589, 89, 680, 815, 915, 216, 904, 513, 109, 27, 103, 832, 875, 959, 892, 214, 570, 278, 835, 259, 71, 342, 457, 296,
# 868, 898, 895, 358, 854, 652, 444, 965, 837, 757, 119, 518, 173, 192, 196, 108, 786, 943, 418, 138, 231, 929, 100, 245, 499, 283, 879, 834, 497, 809, 938, 896, 566, 521, 156, 987, 725, 698, 990, 357, 716, 622, 616, 146, 945, 687, 287, 897, 519, 191, 441, 111, 41, 605, 496, 648, 285, 770, 208, 43, 976, 847, 636, 159, 406, 532, 601, 710, 850, 98, 150, 534, 794, 846, 147, 596, 443, 498, 456, 639, 2, 690, 66, 611, 981, 773, 760, 667, 37, 369, 379, 608, 473, 707, 659, 720, 968, 239, 882, 638, 118, 554, 130, 890, 370, 393, 960, 510, 978, 355, 512, 401, 153, 81, 739, 805, 977, 274, 460, 326, 825, 79, 463, 451, 812, 307, 77, 73, 123, 175, 691, 474, 262, 954, 545, 286, 734, 203, 784, 392, 303, 48, 240, 49, 304, 695, 16, 472, 409, 47, 886, 380, 227, 161, 69, 682, 60, 439, 193, 302, 974, 867, 292, 699, 55, 887, 866, 693, 325, 440, 772, 913, 870, 669, 724, 493, 595, 420, 140, 40, 284, 333, 17, 758, 851, 821, 224, 346, 689, 125, 398, 935, 316, 714, 241, 94, 50, 836, 38, 729, 449, 64, 21, 468, 10, 352, 905, 586, 39, 131, 494, 989, 425, 135, 869, 969, 215, 628, 459, 141, 906, 18, 865, 862, 93, 437, 679, 67, 341, 524, 387, 606, 565, 790, 29, 940, 776, 107, 477, 305, 197, 999, 697, 731, 80, 104, 549, 270, 912, 83, 209, 708, 374, 211, 819, 857, 588, 664, 298, 345, 490, 569, 804, 883, 478, 808, 564, 127, 575, 249, 399, 643, 766, 70, 427, 7, 914, 402, 384, 520, 988, 219, 205, 105, 908, 625, 24, 171, 599, 82, 421, 745,
# 953, 36, 438, 925, 528, 430, 57, 715, 742, 35, 878, 362, 329, 816, 768, 814, 92, 15, 911, 966, 237, 223, 899, 991, 467, 327, 573, 395, 390, 918, 170, 986, 889, 823, 556, 787, 53, 694, 291, 218, 572, 802, 701, 143, 148, 633, 152, 339, 597, 780, 465, 328, 817, 666, 942, 743, 522, 365, 248, 411, 541, 592, 113, 980, 721, 188, 264, 753, 581, 301, 711, 483, 732, 172, 364, 199, 881, 12, 558, 87, 54, 234, 585, 14, 771, 971, 480, 256, 429, 979, 509, 747, 28, 475, 277, 202, 142,
# 272, 765, 856, 931, 859, 685, 754, 948, 604, 619, 376, 426, 651, 46, 946, 764, 389, 930, 961, 576, 920, 626, 136, 306, 568, 717, 221, 557, 404, 381, 308, 860, 555, 853, 884, 32, 22, 801, 356, 683, 615, 507, 621, 255, 656, 613, 169, 74, 315, 806, 391, 843, 653, 594, 226, 260, 132, 85, 662, 201, 359, 179, 99, 267, 928, 598, 86, 893, 271, 23, 72, 383, 275, 641, 52, 655, 242, 529, 350, 9, 348, 984, 90, 962, 424, 120, 166, 157, 290, 206, 590, 602, 624, 750, 228, 59, 709, 416, 792, 712, 752, 863, 888, 849, 247, 783, 178, 514, 788, 874, 511, 657, 186, 324, 803, 505, 607, 78, 155, 432, 145, 686, 257, 403, 254, 855, 880, 185, 373, 560, 6, 649, 332, 730, 251, 957, 949, 533, 217, 117, 225, 181, 762, 671, 807, 195, 934, 8, 375, 630, 347, 445, 65, 30, 591, 759, 300, 947, 926, 84, 423, 45, 531, 408, 629, 748, 723, 993, 388, 487, 164]
#     #print(flag)
#     p = 0
#     for i in range(1000):
#         if i in flag:
#         #if i <= 799:
#             after[p] = after_train_dataset.loc[i].T
#             p = p + 1
#     p = 0
#     for i in range(1000):
#         if i not in flag:
#         #if i >= 800:
#             test[p] = after_train_dataset.loc[i].T
#             p = p + 1
    
#     after_train_dataset = pd.DataFrame(after.values.T, index=after.columns, columns=after.index)
    #after_test_dataset = pd.DataFrame(test.values.T, index=test.columns, columns=test.index)
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
    'SNP45', 'SNP46', 'SNP47', 'SNP48', 'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53']#'SNP54', 'SNP55', 'ACEID']
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

    #print('???????')
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
    #Test_Class_Total = len(after_test_dataset['label'])
    #print(Test_Class_Total)
    #Test_label = after_test_dataset['label']
    print(after_test_dataset)
    Test_Class_Total = 200
    for i in range(Test_Class_Total):
        YES = 1.0
        NO = 1.0
        #o = after_test_dataset.loc[i]
        for item in S_List:
            NORM = 0.0
            v = after_test_dataset[item]
            vo = str(item) + str(' ') + str(v[i])
            #print(vo)
            #print(SenAttribute_yes_count.get(vo, 1))
            YES = YES * SenAttribute_yes_count.get(vo, 1) 
            #print(YES)
            NO = NO * SenAttribute_no_count.get(vo, 1) 
            #print(NO)
            NORM = float(YES + NO)
            YES = YES / NORM
            NO = NO / NORM

        for item in Discrete_Attributes:
            NORM1 = 0.0
            v = after_test_dataset[item]
            vo = str(item) + str(' ') + str(v[i])           
            #print(vo)
            #print(SenAttribute_yes_count.get(vo, 1))
            YES = YES * attribute_yes_count.get(vo, 1)
            #print(YES)
            NO = NO * attribute_no_count.get(vo, 1)
           #print(NO)
            NORM1 = float(YES + NO)
            YES = YES / NORM1
            NO = NO / NORM1

        #print('!!!!!')
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
    #for i in range(Test_Class_Total):
    #    if(pre_label[i] == Test_label[i]):
    #        cnt += 1
    print(pre_label)
    pre = pd.DataFrame(pre_label)
    pre.to_csv('PY_PROJECT/pre.csv', header = 0, index = None, encoding = 'gbk')
    #while 1:
    #    pass
    accuracy = 0
    #accuracy = float(cnt / 200)
    #print('The classify accuracy is: %.2f%%' % (accuracy * 100))
    return accuracy



if __name__ == "__main__":
    result = []
    #for i in range(100):
    after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes = Preprocess()
    acc = Bayers(after_train_dataset, after_test_dataset, Divide_Number, Discrete_Attributes)
    #    result.append(acc)
    #    print(acc)
    #    print(len(rem))
    #    print('====================')
    #    if i == 99:
    #        print(result)
    #        pp = np.argsort(result)
    #        print(pp)
    #        print(result[pp[-1]])
    #        print(rem[pp[-1]])
    #print(max(result))

