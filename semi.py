import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA
import generatemodel as gm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy.linalg as la
from statsmodels.stats.outliers_influence import variance_inflation_factor #用于计算VIF
import perform as pf
from sklearn.cluster import KMeans
import MLE_normal as MN
import scipy.optimize as scio

#f=open('C:/Users/孙永超/Desktop/model/original data/半监督学习—训练集.csv')
#f=open('C:/Users/孙永超/Desktop/model/original data/维信testdata.csv',encoding='UTF-8')
f=open('C:/Users/孙永超/Desktop/model/original data/pf_data_new2.csv',encoding='UTF-8')

#f=open('C:/Users/HP/Desktop/model/original data/testdata.csv')#1200条样本
Data=pd.read_csv(f)
f.close()
#X=Data.iloc[0:10000,:]
#Y=X['scorerevoloan']
#del X['scorerevoloan']
#X_types=pd.Series(X.dtypes,dtype='str')

f2=open('/Users/孙永超/Desktop/model/original data/半监督学习变量.csv',encoding='UTF-8')
missing_value_feed=pd.read_csv(f2)
f2.close()
missing_value_feed.index=missing_value_feed.iloc[:,0]
missing_value_feed=missing_value_feed.iloc[:,2:]

if_letitgo=np.ones([len(Data.columns),1])
for i in  range(0,len(Data.columns)):
    if Data.columns[i] in missing_value_feed.index:
        if_letitgo[i]=0

X=Data.iloc[0:40000,np.where(if_letitgo==0)[0]]
#'''维信'''
#score=X['scorerevoloan']
#del X['scorerevoloan']
#Y_true=X['other_var1']
#del X['other_var1']
'''pf'''
score=X['scorerevoloan']
del X['scorerevoloan']
Y_true=Data.iloc[0:40000,0]




#X = X.iloc[:,np.where((X_types=='float64')|(X_types=='int64'))[0]]

''' 1.粗筛变量，删掉缺失值超过95%（可调整）的变量 '''
nan_ratio_threshold=0.9 #nan_ratio_threshold为阈值
count_null=np.zeros(np.shape(X))
count_null[np.where(X.isnull())]=1
count_null_sumfactor=sum(count_null)/np.shape(X)[0]
count_null_sumfactor=pd.Series(count_null_sumfactor,index=X.columns)
X=X.iloc[:,np.where(count_null_sumfactor<=nan_ratio_threshold)[0]] 


''' 2.删掉非nan同值超过95%（可调整）的变量 '''
mode_ratio_threshold=0.9 #mode_ratio_threshold为阈值
raw_feature_num=len(X.columns)
if_delete_feature=np.zeros([raw_feature_num,1])
for i in range(0,raw_feature_num):
    if_delete_feature[i]=(len(np.where(X.iloc[:,i]==mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])[0][0])[0])/len(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])>mode_ratio_threshold)
X=X.iloc[:,np.where(if_delete_feature==0)[0]]
X_types=pd.Series(X.dtypes,dtype='str')

''' 3.填补空缺值：数值类型变量填充值为-1，类别型变量填充值为blank '''
for i in range(0,np.shape(X)[1]):
    if len(np.where(X.iloc[:,i].isnull())[0])>0: #若有缺失值
        if X_types[i]=='float64' or X_types[i]=='int64':#若为数值型，则填充为指定值           
            if X.columns[i] in missing_value_feed.index:
                X.iloc[np.where(X.iloc[:,i].isnull())[0],i]=missing_value_feed.loc[X.columns[i],'0']                 
        else:#若为分类型，则填充为blank
            X.iloc[np.where(X.iloc[:,i].isnull())[0],i]='blank'
            
#
#for i in range(0,np.shape(X)[1]):
#    if len(np.where(X.iloc[:,i].isna())[0])>0: #若有缺失值
#        if X_types[i]=='float64' or X_types[i]=='int64':#若为数值型，则填充为-99           
#            X.iloc[np.where(X.iloc[:,i].isna())[0],i]=-1           
#            #判断原数据是否是整数型，后面SMOTE也会用到
#            if len(np.where(X.iloc[np.where(~X.iloc[:,i].isna())[0],i]-np.array(X.iloc[np.where(~X.iloc[:,i].isna())[0],i],dtype='int'))[0])==0:
#                X_types[i]='int64'          
#        else:#若为分类型，则填充为blank
#            X.iloc[np.where(X.iloc[:,i].isna())[0],i]='blank'
#            
''' 4.删掉类型超过10（可调整）的非整型非浮点数型变量 '''
if_keep_feature=np.ones([len(X.columns),1]) 
for i in range(0,np.shape(X)[1]):
    if  X_types[i]=='str' or X_types[i]=='object':
         if_keep_feature[i]=1-(len(np.unique(X.iloc[:,i]))>10)#删掉类别变量取值数量大于10的列            
X=X.iloc[:,np.where(if_keep_feature==1)[0]]
X_types=pd.Series(X.dtypes,dtype='str')

'''5.处理类别变量：变为哑变量'''
X_num_position=np.where((X_types=='int64')|(X_types=='float64'))[0]
X_num=X.iloc[:,X_num_position]
X_num_types=X_types[X_num_position]

X_category_position=np.where((X_types=='object')|(X_types=='str'))[0]
X_category=X.iloc[:,X_category_position]
X_category_types=X_types[X_category_position]

X_category_dummies=pd.get_dummies(X_category) #将类型变量变换为哑变量
X_category_dummies_belongs=np.zeros([len(X_category_dummies.columns),1])

count_null_sumfactor_dummys=pd.Series(np.zeros([len(X_category_dummies.columns),]),index=X_category_dummies.columns)

#找到哑变量和原始变量的对应关系
for i in range(0,len(X_category_dummies.columns)):
    for j in range(0,len(X_category.columns)):
        if X_category_dummies.columns[i].find(X_category.columns[j])>=0:
            X_category_dummies_belongs[i]=j
            count_null_sumfactor_dummys.iloc[i]=count_null_sumfactor.loc[X_category.columns[j]]
            break            
            
#X_dummies=X_num.join(X_category_dummies)

X_dummies=pd.concat([X_num,X_category_dummies],axis=1) 
#X_dummies.index=list(range(0,len(X_dummies.index)))         
count_null_sumfactor=count_null_sumfactor.append(count_null_sumfactor_dummys)
X=X_dummies

#X = pd.DataFrame(preprocessing.scale(X),columns=X_dummies.columns)#对其进行归一处理

'''清理相关系数过高的变量组，若两个变量的相关系数大于阈值，删除缺失值高的变量'''
corrcoef_threshold=0.8 #corrcoef_threshold为逐步回归前消除共线性的阈值

#IV_chosen_feature=IV_tot_list.loc[chosen_feature_en_names]
chosen_feature_en_names=list(X.columns)
corrcoef_matrix_chosen_feature=np.corrcoef(X.T)
for i in range(0,corrcoef_matrix_chosen_feature.shape[0]):#将相关系数矩阵对角线元素变为0
    corrcoef_matrix_chosen_feature[i,i]=0

large_in_corrcoef_matrix_chosen_feature=np.where(corrcoef_matrix_chosen_feature>=corrcoef_threshold)#找到相关系数高于阈值的特征位置

#del_times=0
while len(large_in_corrcoef_matrix_chosen_feature[0])>0:#循环操作，直至所有剩余的特征的共线性小于阈值
    first_feature=chosen_feature_en_names[large_in_corrcoef_matrix_chosen_feature[0][0]]
    second_feature=chosen_feature_en_names[large_in_corrcoef_matrix_chosen_feature[1][0]]
    if count_null_sumfactor.loc[first_feature]<=count_null_sumfactor.loc[second_feature]:#对于相关性高于阈值的两个特征，留下缺失值低的，删掉缺失值高的特征
        chosen_feature_en_names.remove(second_feature)
    else:
        chosen_feature_en_names.remove(first_feature)       
    corrcoef_matrix_chosen_feature=np.corrcoef(X.loc[:,chosen_feature_en_names].T)
    for i in range(0,corrcoef_matrix_chosen_feature.shape[0]):
        corrcoef_matrix_chosen_feature[i,i]=0
    large_in_corrcoef_matrix_chosen_feature=np.where(corrcoef_matrix_chosen_feature>=corrcoef_threshold)
    
    #del_times+=1
    #print(del_times)

print('删除相关系数超过{0}的变量后，还剩余{1}个候选变量'.format(corrcoef_threshold,len(chosen_feature_en_names)))
X = X.loc[:,chosen_feature_en_names]

'''6.PCA处理数据'''
X_scaled = X
X_scaled = preprocessing.scale(X)#对其进行归一处理
X_scaled=pd.DataFrame(X_scaled,index=X.index,columns=X.columns)
model=PCA(n_components=0.8, svd_solver='full')
X_pca=model.fit_transform(X_scaled)
X_PCA=pd.DataFrame(X_pca,index=X.index)
#X_pca = preprocessing.scale(X_pca)#对其进行归一处理
print(model.explained_variance_ratio_)
'''Kmeans贴标签'''
#km_model=KMeans(n_clusters=25,
#     init='k-means++', 
#    n_init=100, 
#    max_iter=300, 
#    tol=0.0001, 
#    precompute_distances='auto', 
#    verbose=0, 
#    random_state=714, 
#    copy_x=False, 
#    n_jobs=-1, 
#    algorithm='auto'
#    )
##km_model.fit(np.array(X_scaled))
#km_model.fit(np.array(X_pca))
#lable_pred=pd.DataFrame(km_model.labels_,index=X.index)
#score_mean_new=score
#
#mean_score=[]
#label_sum=[]
#for i in range(0,25):
#    this_label=np.where(lable_pred==i)[0]
#    label_sum.append(len(this_label))
#    mean_score.append(np.mean(score_mean_new.iloc[this_label]))
#
#results=pd.concat([pd.Series(mean_score),pd.Series(label_sum)],axis=1)
'''通过results判断哪些列标记1和0'''
#if_damnchoose=pd.Series(np.zeros([len(score),]),index=X.index)
#if_damnchoose.iloc[np.where((lable_pred.iloc[:,0]==7)|(lable_pred.iloc[:,0]==13)|(lable_pred.iloc[:,0]==15))[0]]=-1#定义为好样本
#if_damnchoose.iloc[np.where((lable_pred.iloc[:,0]==16)|(lable_pred.iloc[:,0]==6))[0]]=1#定义为坏样本
#
#X_with_Y=np.array(X_pca[if_damnchoose!=0])
#X_without_Y=np.array(X_pca[if_damnchoose==0])
#Y=pd.Series(np.zeros([len(X.index),])*np.nan,index=X.index)
#Y.loc[if_damnchoose==1]=1
#Y.loc[if_damnchoose==-1]=0
#Y_X=np.array(Y.loc[if_damnchoose!=0])
#ind1 = len(np.where(Y_X==1))
#ind0 = len(np.where(Y_X==0))
#VIF=[]
#for i in range(0,len(chosen_feature_en_names)):
#   this_VIF=variance_inflation_factor(np.array(X.loc[:,chosen_feature_en_names]),i)
#   VIF.append(this_VIF)
#VIF=pd.Series(VIF,index=chosen_feature_en_names)
#for i in range(len(VIF)):
#    if VIF[i]>10:
#        chosen_feature_en_names.remove(VIF.index[i])
#X = X.loc[:,chosen_feature_en_names]

'''循环聚类'''
ind1 = int(len(X)*0.1)
ind0 = int(len(X)*0.2)

X_Y = X_PCA.join(score).join(Y_true)
X_Y = X_Y.sort_values(by='scorerevoloan')
X_Y_array =np.array(X_Y)
X_with_Y_1 = X_Y_array[0:ind1,:]
X_with_Y_0 = X_Y_array[len(X)-ind0:len(X),:]
X_1 = np.delete(X_with_Y_1,-1,axis=1)
X_0 = np.delete(X_with_Y_0,-1,axis=1)
X_1 = np.delete(X_1,-1,axis=1)
X_0 = np.delete(X_0,-1,axis=1)
X_without_Y = X_Y_array[ind1:len(X)-ind0,:]
X_without_Y = np.delete(X_without_Y,-1,axis=1)
X_without_Y = np.delete(X_without_Y,-1,axis=1)
score_1 = X_with_Y_1[:,-2]
score_0 = X_with_Y_0[:,-2]
Y_true_1 = X_with_Y_1[:,-1]
Y_true_0 = X_with_Y_0[:,-1]
num_1 = ind1
num_0 = ind0

X_without_Y = X_pca[X_Y.index[ind1:len(X)-ind0],:]

while num_1>0.1*ind1:
    print('1')
    km_model_1=KMeans(n_clusters=2,
             init='k-means++', 
             n_init=100, 
             max_iter=300, 
             tol=0.0001, 
             precompute_distances='auto', 
             verbose=0, 
             random_state=714, 
             copy_x=False, 
             n_jobs=-1, 
             algorithm='auto'
             )
    print('1.1')
    km_model_1.fit(np.array(X_1))
    print('1.2')
    lable_pred=km_model_1.labels_
    mean_score=[]
    label_sum=[]
    this_label = []
    for i in range(2):
        this_label += [np.where(lable_pred==i)[0]]
        label_sum.append(len(this_label[i]))
        mean_score.append(np.mean(score_1[this_label[i]]))
    print(label_sum)
    print(mean_score)
    choose_class = np.argmin(mean_score)
    num_1 = label_sum[choose_class]
    if num_1 < 0.05*ind1:
        choose_class = np.argmax(mean_score)
        num_1 = label_sum[choose_class]
    X_without_Y = np.vstack([X_without_Y,X_1[this_label[1-choose_class]]])
    X_1 = X_1[this_label[choose_class],:]
    score_1 = score_1[this_label[choose_class]]
    Y_true_1 = Y_true_1[this_label[choose_class]]
    
    
while num_0>0.1*ind0:
    km_model_0=KMeans(n_clusters=2,
             init='k-means++', 
             n_init=100, 
             max_iter=300, 
             tol=0.0001, 
             precompute_distances='auto', 
             verbose=0, 
             random_state=714, 
             copy_x=False, 
             n_jobs=-1, 
             algorithm='auto'
             )
    km_model_0.fit(np.array(X_0))
    lable_pred=km_model_0.labels_
    mean_score=[]
    label_sum=[]
    this_label = []
    for i in range(2):
        this_label += [np.where(lable_pred==i)[0]]
        label_sum.append(len(this_label[i]))
        mean_score.append(np.mean(score_0[this_label[i]]))
    print(label_sum)
    print(mean_score)
    choose_class = np.argmax(mean_score)
    num_0 = label_sum[choose_class]
    if num_0 < 0.05*ind0:
        choose_class = np.argmin(mean_score)
        num_0 = label_sum[choose_class]
    X_without_Y = np.vstack([X_without_Y,X_0[this_label[1-choose_class]]])
    X_0 = X_0[this_label[choose_class],:]
    score_0 = score_0[this_label[choose_class]]
    Y_true_0 = Y_true_0[this_label[choose_class]]
    

X_with_Y = np.vstack([X_1,X_0])
Y_X = np.vstack([np.ones([num_1,1]),np.zeros([num_0,1])])

'''7.给打分前10%和后10%（可调整）的样本打上0和1的标签'''
#ind1 = int(len(X)*0.05)
#ind0 = int(len(X)*0.2)
#Y_X = np.hstack((np.ones(ind1),np.zeros(ind0)))
#X_Y = X.join(score).join(Y_true)
#X_Y = X_Y.sort_values(by='scorerevoloan')
#X_with_Y = np.vstack((X_pca[X_Y.index[0:ind1],:],X_pca[X_Y.index[len(X)-ind0:len(X)],:]))
#X_without_Y = X_pca[X_Y.index[ind1:len(X)-ind0],:]




[Miu,Sigma,a] = gm.train(X_with_Y,Y_X,X_without_Y,0.01)
sig = []
W = []
for i in range(2):
    b,c = la.eig(Sigma[i])
    sig += [la.norm(np.diag(c.dot(np.sqrt(b))),axis=1)]
    W += [np.diag(np.sqrt(1/b)).dot(c.T)]
pred = []
score_pred = []

#for i in range(ind1):
#    pred += [gm.predict(X_with_Y[i],Miu,Sigma,a)[0]]
#    score_pred += [gm.predict(X_with_Y[i],Miu,Sigma,a)[1]]
#    
#for i in range(len(X)):
#    score_pred += [gm.predict(X_pca[i,:],Miu,Sigma,a)[1]]

for i in range(len(X)):
#    Pred,Score_pred = gm.predict2(X_pca[i,:],Miu,a,W,sig,np.inf)
    Pred,Score_pred = gm.predict(X_pca[i,:],Miu,Sigma,a)
    pred += [Pred]
    score_pred += [Score_pred]
#for i in range(len(X)):
#    pred += [gm.predict2(X_pca[i,:],Miu,a,W)[1]]
score_pred = np.array(score_pred)
score_pred[np.where(score_pred<0)] = 0
ks_value,bad_percent,good_percent=pf.cal_ks(-score_pred,Y_true,section_num=20)
for i in range(num_0):
    Pred,Score_pred = gm.predict2(X_with_Y[i+num_1],Miu,a,W,sig,np.inf)
    pred += [Pred]
    score_pred += [Score_pred]
#score = np.array(score)
plt.plot(score_pred,'.')
#plt.plot(pred,'.')
##pca = PCA()
#pca.fit(X)
#pca.components_
#'''极大似然估计，无监督'''
#X_plus = np.hstack([X_pca[:,[0,1]],np.ones([len(X_pca),1])])
#def like(A):
#    return(-MN.likelihood(A,X_plus))
#def dlike(A):
#    return(-np.array(MN.dlikelihood(A,X_plus)).reshape(len(X_plus.T),1))
#A0 = np.ones([np.shape(X_plus)[1],1])
#OPT = scio.minimize(like,A0,method='Nelder-Mead',tol=0.01)
#A = OPT.x
#score_un = X_plus.dot(A)
#ks_value2,bad_percent2,good_percent2=pf.cal_ks(-score_un,Y_true,section_num=20)
