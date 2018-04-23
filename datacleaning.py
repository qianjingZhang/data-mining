# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:23:35 2017

@author: 爱做梦的张猪猪
"""
import pandas as pd
import numpy as np
import xlrd
import gc
#寻找最近K个点
def find_nearest_k(col,cols,w_value,k):
    #寻找该值附近的可能点
    w_ind = cols.index(w_value) #目标点下标
    low_ind = max(0,w_ind-k)
    high_ind = min(len(cols),w_ind+k)
    w_likely = cols[int(low_ind):int(high_ind)]
    #计算距离
    w_dis = []
    for w_n in range(len(w_likely)):
        w_dis.append([w_n, abs(w_likely[w_n]-w_value)])
    #距离排序
    w_dis_o = sorted(w_dis, key = lambda item: item[1])
    #获取k点值
    w_k = []
    for w_n in w_dis_o[:k]:
        w_k.append(w_likely[w_n[0]])
    return w_k
# the preprocess of raw data
workbook = xlrd.open_workbook('F:\\NewStudents.xlsx','r')
booksheet = workbook.sheet_by_index(0)
#cell_11=booksheet.cell_value(0,0)
#row_3 = booksheet.row_values(2)
cols = booksheet.ncols #列数
rows = booksheet.nrows #获取行数
print(rows)
print(cols)
final_ind = booksheet.cell_value(rows-1,0)#最后的那个序号
j=0 #用来记录序号后延的位置
thre =2 #检测是否为离群点
thre_v = 2 #单特征汇总后离群点删除的阈值
#按性别进行分类，方便下述具有性别特质的特质处理计算
data_fel = []
data_male = []
for r in range(rows):
    row = booksheet.row_values(r)
    gender = row[1]
    if gender == 0:
        data_male.append(row)
    elif gender == 1:
        data_fel.append(row)
data_male = np.array(data_male)
data_fel = np.array(data_fel)
#对特征进行筛选
#如果在项目名称列出现空缺值很大程度上是该列是不应该存在的数据
#为防止误删数据，对该列提取并判断其空缺值占比,大则直接删除
#判断项目中是否存在与性别或者年龄相关的项目
#本次中不存在
#缺失值处理，高斯分布补充的方式
list_name = booksheet.row_values(0)
name = np.array(list_name)
#print(name)
list_dmis_male = []
list_dmis_fel = []
for c in range(len(list_name)):
    title = list_name[c]
    col = booksheet.col_values(c)
    col_male = np.array(data_male[:,c])
    col_fel = np.array(data_fel[:,c])   
    mis = col.count('')
    if mis <0.2*len(col):
       col_male = col_male.astype(float)
       col_fel = col_fel.astype(float)
       miu_fel = col_fel.mean()
       std_fel = col_fel.std()
       miu_male = col_male.mean()
       std_male = col_male.std()   
       #高斯分布处理缺失值
       for w in range(len(col_male)):
           if col_male[w] == '':
              col_male[w] = np.random.normal(miu_male,std_male,1)
       for w in range(len(col_fel)):
           if col_fel[w] == '':
              col_fel[w] = np.random.normal(miu_fel,std_fel,1)
       list_dmis_fel.append(col_fel)
       list_dmis_male.append(col_male)
del data_fel
del data_male
#寻找错误输入数据与重复数据
list_dmis_fel = np.array(list_dmis_fel)
data_fel = list_dmis_fel.T
print("缺失值处理后",np.size(data_fel,0),np.size(data_fel,1))
del list_dmis_fel
list_dmis_male = np.array(list_dmis_male)
data_male = list_dmis_male.T
print("缺失值处理后",np.size(data_male,0),np.size(data_male,1))
del list_dmis_male
#判断重复数据以及编号错误
fel_index = data_fel[:,0]
male_index = data_male[:,0]
data_male_uni = []
data_fel_uni = []
#删除女性部分重复数据
for ind_f in range(len(fel_index)-1):
    ind_1 = fel_index[ind_f]
    ind_2 = fel_index[ind_f+1]
    if ind_1 ==ind_2:
        data_fel_1 = data_fel[ind_f,:]
        data_fel_2 = data_fel[ind_f+1,:]
        total =0
        for c in range(len(data_fel_1)):
            total = total + float(data_fel_1[c])-float(data_fel_2[c])
        aver = total/len(data_fel_1)
        if aver > 1:#差距过大依旧保留,说明该数据并非重复数据而是编号有误，此时修改编号（相当于往后插入）
           data_fel_uni.append(ind_f)
           j=j+1
           data_fel[ind_f,0]=final_ind+j #序号改为当前最大序号值+1
    else:
        data_fel_uni.append(ind_f)
data_fel = data_fel[data_fel_uni,:]
print("重复数据删除后",np.size(data_fel,0),np.size(data_fel,1))
del data_fel_uni
#删除男性部分重复数据
for ind_f in range(len(male_index)-1):
    ind_1 = male_index[ind_f]
    ind_2 = male_index[ind_f+1]
    if ind_1 ==ind_2:
        data_male_1 = data_male[ind_f,:]
        data_male_2 = data_male[ind_f+1,:]
        total =0
        for c in range(len(data_male_1)):
            total = total + float(data_male_1[c])-float(data_male_2[c])
        aver = total/len(data_male_1)
        if aver > 1:#差距过大依旧保留
           data_male_uni.append(ind_f)
           j=j+1
           data_male[ind_f,0]=final_ind+j
    else:
        data_male_uni.append(ind_f)
data_male = data_male[data_male_uni,:]
print("重复数据删除后",np.size(data_male,0),np.size(data_male,1))
del data_male_uni
#查找离群点和奇异点
#考虑到同一条数据有多个特征，所以采用标记的方式,单条数据如果出现2个奇异点则删除
#女性标记,单属性筛选
m = np.size(data_fel,0)
n = np.size(data_fel,1)
fel_mark = np.zeros([m,n])
fel_mark[:,0] = data_fel[:,0]
k=3 #确定考虑点个数
for c in range(1,n):
    col = data_fel[:,c]
    #对数据按大小进行排序
    cols = sorted(col)
    for w in range(len(col)):
        #计算离群点
        #获取目标点的值
        w_value=col[w]
        #寻找最近的K个点
        w_k=find_nearest_k(col,cols,w_value,k)
        #计算distance(A,B),记录Nk(A)以及Nk(B)
        #此处进行一些简化，默认Nk(A)以及Nk(B)始终为k，实际上可能会大于k
        totA =0
        totB_all=[]
        totB = 0
        for w_k_v in w_k:#  A的K近邻即B
            #A的k近邻点B的近邻点C，计算distancek(B)
            w_k_B = find_nearest_k(col,cols,w_k_v,k)
            k_dis = np.sqrt((w_k_B[2]-w_k_v)**2)
            dis = max(k_dis,np.sqrt((w_k_v-w_value)**2))
            totA = totA + dis
            for w_k_n in w_k_B:
                #寻找C的k近邻点，计算distancek(C)
                w_k_B_d = find_nearest_k(col,cols,w_k_n,k)
                #计算distancek(C)
                k_dis = np.sqrt((w_k_B_d[2]-w_k_n)**2)
                #max(distancek(C),distance(B,C))
                dis = max(k_dis,np.sqrt((w_k_v -w_k_n)**2))
                totB = totB+dis
            if totB ==0:
               totB = 0.000000001
            totB_all.append(totB)
            tot_B = 0
        #计算lrdA以及lrdB
        #避免出现nan，对totA进行处理
        if totA ==0:
           totA = 0.000000001
        lrdA=1/(totA/k)
        LOFA = 0
        for v in totB_all:
            lrdB = 1/(v/k)
            LOFA = LOFA+ lrdB/lrdA
        LOFA = LOFA/k
        print(LOFA)
        #如果值大于1，说明离群点
        if LOFA >thre:
            fel_mark[w,c]=1
            print("error")
#对奇异点数据进行删除
num_outler = np.sum(fel_mark[:,1:n-1], 1)#序号那一列不加入计算
data_fel_uni=[]
for w in range(len(num_outler)):
    if num_outler[w]<thre_v:
        data_fel_uni.append(w)
data_fel = data_fel[data_fel_uni,:]
print("单特征计算奇异点删除后",np.size(data_fel,0),np.size(data_fel,1))
del fel_mark
del data_fel_uni
del totB_all
#对个例进行整体的奇异值判断
#对于所有特征值按行求和
fel_mark = np.sum(data_fel[:,1:n-1],1) #此处用以记载所有样本的单样本所有特征之和
col = fel_mark
cols = sorted(col)
LOFA_all = []
for w in range(len(col)):
    w_value = col[w]
    w_k = find_nearest_k(col,cols,w_value,k)
    totA =0
    totB_all=[]
    totB=0
    for w_k_v in w_k:
        w_k_B =find_nearest_k(col,cols,w_k_v,k)
        k_dis = np.sqrt((w_k_B[2]-w_k_v)**2)
        dis = max(k_dis,np.sqrt((w_k_v-w_value)**2))
        totA =totA +dis
        for w_k_n in w_k_B:
            w_k_B_d = find_nearest_k(col,cols,w_k_n,k)
            k_dis = np.sqrt((w_k_B_d[2]-w_k_n)**2)
            dis = max(k_dis, np.sqrt((w_k_v-w_k_n)**2))
            totB =totB +dis
        if totB ==0:
            totB = 0.000000001
        totB_all.append(totB)
        totB = 0
    if totA ==0:
       totA = 0.000000001
    lrdA = 1/(totA/k)
    LOFA =0
    for v in totB_all:
        lrdB = 1/(v/3)
        LOFA = LOFA +lrdB/lrdA
    LOFA = LOFA/k
    print(LOFA)
    if LOFA < thre:
       LOFA_all.append(w)
#删除奇异点
data_fel = data_fel[LOFA_all,:]
del fel_mark
del LOFA_all
del totB_all
print("整体特征计算奇异点删除后",np.size(data_fel,0),np.size(data_fel,1))
#男性标记
m = np.size(data_male,0)
n = np.size(data_male,1)
male_mark = np.zeros([m,n])
male_mark[:,0] = data_male[:,0]
k=3 #确定考虑点个数
for c in range(1,n):
    col = data_fel[:,c]
    #对数据按大小进行排序
    cols = sorted(col)
    for w in range(len(col)):
        #计算离群点
        #获取目标点的值
        w_value=col[w]
        #寻找最近的K个点
        w_k=find_nearest_k(col,cols,w_value,k)
        #计算distance(A,B),记录Nk(A)以及Nk(B)
        #此处进行一些简化，默认Nk(A)以及Nk(B)始终为k，实际上可能会大于k
        totA =0
        totB_all=[]
        totB = 0
        for w_k_v in w_k:#  A的K近邻即B
            #A的k近邻点B的近邻点C，计算distancek(B)
            w_k_B = find_nearest_k(col,cols,w_k_v,k)
            k_dis = np.sqrt((w_k_B[2]-w_k_v)**2)
            dis = max(k_dis,np.sqrt((w_k_v-w_value)**2))
            totA = totA + dis
            for w_k_n in w_k_B:
                #寻找C的k近邻点，计算distancek(C)
                w_k_B_d = find_nearest_k(col,cols,w_k_n,k)
                #计算distancek(C)
                k_dis = np.sqrt((w_k_B_d[2]-w_k_n)**2)
                #max(distancek(C),distance(B,C))
                dis = max(k_dis,np.sqrt((w_k_v -w_k_n)**2))
                totB = totB+dis
            if totB ==0:
                totB = 0.000000001
            totB_all.append(totB)
            tot_B = 0
        #计算lrdA以及lrdB
        #避免出现nan，对totA进行处理
        if totA ==0:
            totA = 0.000000001
        lrdA=1/(totA/k)
        LOFA = 0
        for v in totB_all:
            lrdB = 1/(v/k)
            LOFA = LOFA+ lrdB/lrdA
        LOFA = LOFA/k
        #如果值大于1，说明离群点
        if LOFA >thre:
            male_mark[w,c]=1
            print("error")

#对奇异点数据进行删除
num_outler = np.sum(male_mark[:,1:n-1], 1)
data_male_uni=[]
for w in range(len(num_outler)):
    if num_outler[w]<thre_v:
        data_male_uni.append(w)
data_male = data_male[data_male_uni,:]
print("单个特征奇异点删除后",np.size(data_male,0),np.size(data_male,1))
del male_mark
del totB_all
del data_male_uni
#整体特征计算
#对于所有特征值按行求和
male_mark = np.sum(data_male[:,1:n-1],1) #此处用以记载所有样本的单样本所有特征之和
col = male_mark
cols = sorted(col)
LOFA_all = []
for w in range(len(col)):
    w_value = col[w]
    w_k = find_nearest_k(col,cols,w_value,k)
    totA =0
    totB_all=[]
    totB=0
    for w_k_v in w_k:
        w_k_B =find_nearest_k(col,cols,w_k_v,k)
        k_dis = np.sqrt((w_k_B[2]-w_k_v)**2)
        dis = max(k_dis,np.sqrt((w_k_v-w_value)**2))
        totA =totA +dis
        for w_k_n in w_k_B:
            w_k_B_d = find_nearest_k(col,cols,w_k_n,k)
            k_dis = np.sqrt((w_k_B_d[2]-w_k_n)**2)
            dis = max(k_dis, np.sqrt((w_k_v-w_k_n)**2))
            totB =totB +dis
        if totB ==0:
            totB = 0.000000001
        totB_all.append(totB)
        totB = 0
    if totA ==0:
       totA = 0.000000001
    lrdA = 1/(totA/k)
    LOFA =0
    for v in totB_all:
        lrdB = 1/(v/3)
        LOFA = LOFA +lrdB/lrdA
    LOFA = LOFA/k
    if LOFA < thre:
       LOFA_all.append(w)
#删除奇异点
data_male = data_male[LOFA_all,:]
del male_mark
del LOFA_all
del totB_all
print("整体特征计算奇异点删除后",np.size(data_male,0),np.size(data_male,1))

#数据合并
data = np.vstack((data_male,data_fel))
print("最终数据",np.size(data,0),np.size(data,1))
n = np.size(data,1)
#数据保存

name = list_name #定义表头
data1 = pd.DataFrame(data[:,1:n],columns = list_name[1:38])
#data1.rename_axis(list_name, axis =1)

del data
data1.to_excel('D:\\data1.xlsx')
data1.to_csv('D:\\data1.csv')
del data_male
del data_fel
del booksheet
gc.collect()
