import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA

def multinormal_pdf(x,mu,sigma):
    dim = len(x.T)
    det_sig = la.det(sigma)
    part1 = 1 / ( ((2* np.pi)**(dim/2)) * (det_sig**(1/2)) )
#    part2 = (-1/2) * ((x-mu).T.dot(la.inv(sigma))).dot((x-mu))
    part2 = (-1/2) * np.sum((((x-mu).dot(la.inv(sigma)))*((x-mu))).T,axis=0)
    pdf = part1 * np.exp(part2)
    return(pdf)
    
def normal_pdf(x,mu,sigma,tol=0.1):
#    sigma_use=sigma(np.where(sigma>tol))
    dim = len(x.T)
    part1 = 1 / ( ((2* np.pi)**(dim/2)) * (np.prod(sigma)**(1/2)) )
    part2 = (-1/2) * np.sum((((x-mu)**2/sigma)).T,axis=0)
    pdf = part1 * np.exp(part2)
    return(pdf)
    
def train(X,Y,X_withoutY,tol):
    num = X.shape[0]
    num2 = X_withoutY.shape[0]
    dim = X.shape[1]
    number_of_class = len(np.unique(Y))
    X_classed = []
    PCA_model = []
    X_withoutY_list = []
    for i in range(number_of_class):
        X_classed += [X[np.where(Y==i)[0],:]]
        PCA_model += [PCA(n_components=0.8, svd_solver='full')]
        X_classed[i] = PCA_model[i].fit_transform(X_classed[i])
        X_withoutY_list += [PCA_model[i].fit_transform(X_withoutY)]
    miu = [np.mean(x,axis=0) for x in X_classed]
    Sigma = [np.cov(x, rowvar=False) for x in X_classed]
    a = np.array([len(X_classed[i])/(num) for i in range(number_of_class)])
    dmiu = miu
    dSigma =Sigma
    tol_miu = tol*max([la.norm(M) for M in dmiu])
    tol_Sigma = tol*max([la.norm(S) for S in dSigma])        
    p = 0
    
    while (max([la.norm(M) for M in dmiu]) > tol_miu) | (max([la.norm(S) for S in dSigma]) > tol_Sigma):
        pdf = []
        gamma = []
        miu_new = []
        Sigma_new = []
        a_new = []
        p += 1
        print(p)
        sig=[]
        W=[]
        for i in range(number_of_class):
            pdf += [multinormal_pdf(X_withoutY_list[i],miu[i],Sigma[i])*a[i]]
#            b,c = la.eig(Sigma[i])
#            sig += [np.diag(la.norm(np.diag(c.dot(np.sqrt(b))),axis=1))]
#            W += [np.diag(np.sqrt(1/b)).dot(c.T)] 
##            pdf += [multinormal_pdf(((X_withoutY-miu[i]).dot(W[i].T))[:,np.where(sig[i]>0.1)[0]],np.zeros(len(np.where(sig[i]>0.1)[0])),np.eye(len(np.where(sig[i]>0.1)[0])))*a[i]]
#            pdf += [multinormal_pdf((X_withoutY-miu[i]).dot(W[i].T),np.zeros(dim),np.eye(dim))*a[i]]

        pdf = np.array(pdf)
        sumpdf = np.sum(pdf,axis=0)
        sumpdf[np.where(sumpdf==0)[0]]=1
        for i in range(number_of_class):
            gamma += [(pdf[i,:]+1/num2)/(sumpdf+2/num2)]
#            gamma += [(pdf[i,:])/(sumpdf)]
            miu_new += [(np.sum(np.array([gamma[i]]*dim).T*X_withoutY_list[i],axis=0)+np.sum(X_classed[i],axis=0))/(np.sum(gamma[i])+len(X_classed[i]))]
            mat1 = [gamma[i][j]*np.mat(X_withoutY_list[i][j]-miu[i]).T.dot(np.mat(X_withoutY_list[i][j]-miu[i]))for j in range(num2)]   
            matrix1 = np.zeros([dim,dim])
            for mat in mat1:  
                matrix1 = matrix1 + mat
                matrix1 = np.array(matrix1)
            mat2 = [np.mat(x_classed-miu[i]).T.dot(np.mat(x_classed-miu[i]))for x_classed in X_classed[i]]   
            matrix2 = np.zeros([dim,dim])
            for mat in mat2:  
                matrix2 = matrix2 + mat
                matrix2 = np.array(matrix2)
            Sigma_new += [(matrix1+matrix2)/(np.sum(gamma[i])+len(X_classed[i]))]
            a_new += [(np.sum(gamma[i])+len(X_classed[i]))/(num+num2)]
            dmiu[i] = miu_new[i] - miu[i]
            dSigma[i] = Sigma_new[i] - Sigma[i]
        miu = miu_new
        Sigma = Sigma_new
        a = a_new
    return(miu,Sigma,a,PCA_model)
    
    
def predict(X,Miu,Sigma,a):
    n = len(Miu)
    pdf = []
    for i in range(n):
        pdf += [multinormal_pdf(X,Miu[i],Sigma[i])*a[i]]
    pdf = np.array(pdf)
    if sum(pdf)==0:
        score = -1
        type_predict = -1
    else:
        score = pdf[1]/sum(pdf)
        type_predict = np.argmax(pdf)
    return(type_predict,score)
    
def predict2(X,Miu,a,W,sig,num):
    n = len(Miu)
    m = len(X)
    pdf = []
    for i in range(n):
        pdf += [normal_pdf((X-Miu[i]).dot(W[i].T),np.zeros(m),np.ones(m))*a[i]]
#        pdf += [multinormal_pdf(((X-Miu[i]).dot(W[i].T))[np.where(sig[i]>0.1)[0]],np.zeros(len(np.where(sig[i]>0.1)[0])),np.eye(len(np.where(sig[i]>0.1)[0])))*a[i]]
#        pdf += [multinormal_pdf((X-Miu[i]).dot(W[i].T),np.zeros(m),np.eye(m))*a[i]]
    pdf = np.array(pdf)
    if sum(pdf)+2/num==0:
        score = -1
        type_predict = -1
    else:
        score = (pdf[1]+1/num)/(sum(pdf)+2/num)
        type_predict = np.argmax(pdf)
    return(type_predict,score)