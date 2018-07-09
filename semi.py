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

f=open('C:/Users/Desktop/model/original data/半监督学习—训练集.csv')


#f=open('C:/Users/HP/Desktop/model/original data/testdata.csv')#1200条样本
Data=pd.read_csv(f)
f.close()
#X=Data.iloc[0:10000,:]
#Y=X['scorerevoloan']
#del X['scorerevoloan']
#X_types=pd.Series(X.dtypes,dtype='str')

f2=open('/Users/Desktop/model/original data/半监督学习变量.csv',encoding='UTF-8')
missing_value_feed=pd.read_csv(f2)
f2.close()
missing_value_feed.index=missing_value_feed.iloc[:,0]
missing_value_feed=missing_value_feed.iloc[:,2:]

if_letitgo=np.ones([len(Data.columns),1])
for i in  range(0,len(Data.columns)):
    if Data.columns[i] in missing_value_feed.index:
        if_letitgo[i]=0

X=Data.iloc[0:40000,np.where(if_letitgo==0)[0]]

'''pf'''
score=X['scorerevoloan']
del X['scorerevoloan']
Y_true=Data.iloc[0:40000,0]




#X = X.iloc[:,np.where((X_types=='float64')|(X_types=='int64'))[0]]


''' 粗筛变量，删掉缺失值超过95%（可调整）的变量 '''
nan_ratio_threshold=0.9 #nan_ratio_threshold为阈值
count_null=np.zeros(np.shape(X))
count_null[np.where(X.isnull())]=1
count_null_sumfactor=sum(count_null)/np.shape(X)[0]
count_null_sumfactor=pd.Series(count_null_sumfactor,index=X.columns)
X=X.iloc[:,np.where(count_null_sumfactor<=nan_ratio_threshold)[0]] 

''' 删掉非nan同值超过95%（可调整）的变量 '''
mode_ratio_threshold=0.9 #mode_ratio_threshold为阈值
raw_feature_num=len(X.columns)
if_delete_feature=np.zeros([raw_feature_num,1])
for i in range(0,raw_feature_num):
    if_delete_feature[i]=(len(np.where(X.iloc[:,i]==mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])[0][0])[0])/len(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])>mode_ratio_threshold)
X=X.iloc[:,np.where(if_delete_feature==0)[0]]
X_types=pd.Series(X.dtypes,dtype='str')

''' 填补空缺值：数值类型变量填充值为指定值，类别型变量填充值为blank '''
for i in range(0,np.shape(X)[1]):
    if len(np.where(X.iloc[:,i].isnull())[0])>0: #若有缺失值
        if X_types[i]=='float64' or X_types[i]=='int64':#若为数值型，则填充为指定值           
            if X.columns[i] in missing_value_feed.index:
                X.iloc[np.where(X.iloc[:,i].isnull())[0],i]=missing_value_feed.loc[X.columns[i],'0']                 
        else:#若为分类型，则填充为blank
            X.iloc[np.where(X.iloc[:,i].isnull())[0],i]='blank'
            
''' 删掉类型超过10（可调整）的非整型非浮点数型变量 '''
if_keep_feature=np.ones([len(X.columns),1]) 
for i in range(0,np.shape(X)[1]):
    if  X_types[i]=='str' or X_types[i]=='object':
         if_keep_feature[i]=1-(len(np.unique(X.iloc[:,i]))>10)#删掉类别变量取值数量大于10的列            
X=X.iloc[:,np.where(if_keep_feature==1)[0]]
X_types=pd.Series(X.dtypes,dtype='str')

'''处理类别变量：变为哑变量'''
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

X_dummies=pd.concat([X_num,X_category_dummies],axis=1)     
count_null_sumfactor=count_null_sumfactor.append(count_null_sumfactor_dummys)
X=X_dummies


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

'''PCA处理数据'''
X_scaled = X
X_scaled = preprocessing.scale(X)#对其进行归一处理
X_scaled=pd.DataFrame(X_scaled,index=X.index,columns=X.columns)
model=PCA(n_components=0.8, svd_solver='full')
X_pca=model.fit_transform(X_scaled)
X_PCA=pd.DataFrame(X_pca,index=X.index)
print(model.explained_variance_ratio_)

'''循环聚类，通过score和聚类给部分样本打标签'''
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
             n_jobs=1, 
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
             n_jobs=1, 
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

'''分别对两类数据X_1和X_2做pca，不能调包，需要找出特征值比较小的方向'''
eigenvalue_0,featurevector_0=la.eig(np.cov(X_0.T))
eigenvalue_1,featurevector_1=la.eig(np.cov(X_1.T))
idx_0 = eigenvalue_0.argsort()[::-1]
idx_1 = eigenvalue_1.argsort()[::-1]
eigenValues_0 = eigenvalue_0[idx_0]
eigenValues_1 = eigenvalue_1[idx_1]
FeatureVectors_0 = featurevector_0[:,idx_0]
FeatureVectors_1 = featurevector_1[:,idx_1]

'''设置阈值alpha，对两组数据中方差小于alpha的方向做逻辑回归，对两组数据中方差大于alpha的方向做高斯混合模型'''
alpha = 0.4
LR_Vectors = np.vstack([FeatureVectors_0[eigenValues_0<alpha,:],FeatureVectors_1[eigenValues_1<alpha,:]])
GM_Vectors = np.vstack([FeatureVectors_0[eigenValues_0>alpha,:],FeatureVectors_1[eigenValues_1>alpha,:]])
X_Y_LR = X_with_Y.dot(LR_Vectors.T)
X_Y_GM = X_with_Y.dot(GM_Vectors.T)
X_LR = X_pca.dot(LR_Vectors.T)
X_GM = X_pca.dot(GM_Vectors.T)
X_without_Y_GM = X_without_Y.dot(GM_Vectors.T)

'''再做一次PCA'''
PCA_LR=PCA(n_components=0.8, svd_solver='full')
X_Y_LR_pca=model.fit_transform(X_Y_LR)
X_LR_pca=model.transform(X_LR)

PCA_GM = PCA(n_components=0.8, svd_solver='full')
X_Y_GM_pca = model.fit_transform(X_Y_GM)
X_GM_pca = model.transform(X_GM)
X_without_Y_GM_pca = model.transform(X_without_Y_GM)


'''逻辑回归'''
LR_model = LogisticRegression(penalty='l1', 
                              dual=False, 
                              tol=0.0001, 
                              C=1.0, 
                              fit_intercept=True, 
                              intercept_scaling=1, 
                              class_weight=None, 
                              random_state=None, 
                              solver='saga', 
                              max_iter=200, 
                              multi_class='ovr', 
                              verbose=0, 
                              warm_start=False, 
                              n_jobs=1
                              )
LR_model.fit(np.hstack([X_Y_LR]), Y_X)
score_pred_LR = LR_model.decision_function(np.hstack([X_LR]))
ks_value_LR,bad_percent_LR,good_percent_LR=pf.cal_ks(-score_pred_LR,Y_true,section_num=20)

'''高斯混合模型'''
[Miu,Sigma,a] = gm.train(X_Y_GM,Y_X,X_without_Y_GM,0.001)
score_pred_GM = gm.predict(X_GM,Miu,Sigma,a)
#score_pred_GM = gm.predict3(X_GM,Miu,Sigma,a)
score_pred_GM[np.where(score_pred_GM<0)] = 0
ks_value_GM,bad_percent_GM,good_percent_GM=pf.cal_ks(-score_pred_GM,Y_true,section_num=20)

'''将两个结果标准化再按比例相加'''
score_pred_LR = preprocessing.scale(score_pred_LR)
score_pred_GM = preprocessing.scale(score_pred_GM)
score_pred = score_pred_LR + score_pred_GM
ks_value,bad_percent,good_percent=pf.cal_ks(-score_pred,Y_true,section_num=20)
