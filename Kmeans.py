import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import calinski_harabasz_score as CH
from sklearn.metrics import adjusted_rand_score as acc
from tqdm import tqdm


#%%  PCA

def PCA (x_1):
    x_1 = load_digits().data    
    mean = np.average(x_1,axis=0)

    x = (x_1-mean)/np.std(x_1)
    A = (1/len(x))*np.dot(np.transpose(x),x)

    U, L, Ut = np.linalg.svd(A)

    rank=0
    Uk = U[:,:rank]
    Lk = L[:rank]

    while  (np.sum(Lk)/np.sum(L)) <= 0.9:
        rank+=1
        Uk = U[:,:rank]
        Lk = L[:rank]
    
    xk = np.dot(x,Uk)

    return xk

#%%   k means clustering


def kmeans_clustering (samples,k,itr,withPCA,random_init):
    
    if withPCA:
        samples = PCA(samples)
    else: 
        samples = (samples-np.average(samples,axis=0))/np.std(samples)
        # samples = samples
    
    ### initializing centroids
    
    rindex = []
    
    for i in range(k):
        r = np.random.randint(1,len(samples))
        rindex.append(r)
    
    rindex_fixed = [17, 80, 70, 125, 89, 38, 109, 119, 18, 32]
    
    if random_init:
        rindex = rindex
    else:
        rindex = rindex_fixed
    
        
    mu = samples[rindex,:]
    L2norms = np.empty(len(samples))
    temp_norm = np.empty(len(samples))
    CH_index = []
    J = []
    
    ### Finding distances and labels
    
    for m in range(0,len(mu)):
          for n in range(0,len(samples)):
                temp_norm[n] = np.sqrt(np.sum((samples[n,:]-mu[m,:])**2))
          L2norms = np.vstack((L2norms,temp_norm))
    L2norms = np.delete(L2norms,0,0).T
    
    c = np.array([np.argmin(a) for a in L2norms])
    
    
    ### Updating centroids and labels
    
    for m in tqdm(range(itr)):
        mu = []
        for n in range(k):
            temp_mu = samples[c==n].mean(axis=0)
            mu.append(temp_mu)
        mu = np.vstack(mu)
        
        L2norms = np.empty(len(samples))
        temp_norm = np.empty(len(samples))
        
        for u in range(0,len(mu)):
              for v in range(0,len(samples)):
                    temp_norm[v] = np.sqrt(np.sum((mu[u,:]-samples[v,:])**2))
              L2norms = np.vstack((L2norms,temp_norm))
        L2norms = np.delete(L2norms,0,0).T
        
        c = np.array([np.argmin(b) for b in L2norms])
        
        ### Calculating loss
        
        J_temp = np.mean(np.square(np.linalg.norm(samples-mu[c], axis=1)))
        J = np.append(J,J_temp)
    
    J_avg = np.mean(J)
    CH_index.append(CH(samples,c))
    
    return c,J_avg,CH_index

X = load_digits().data
g = load_digits().target

labels, loss, CH_score = kmeans_clustering(X,10,50,1,1)
sim = acc(g,labels)
# X1 = PCA(X)
# u_labels = np.unique(labels)
# for i in u_labels:
#     plt.scatter(X1[labels == i , 0] , X1[labels == i , 1],label=i)
# plt.legend()
# plt.show()

# ratio = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# no_of_clusters = [2,3,4,5,6,7,8,9,10]


# ### varying number of samples

# losses = []
# CH_mat = []
# sim_mat = []
# for i in ratio:
#     x_train,x_other = train_test_split(X,train_size=i, shuffle=False)
#     labels, loss, CH_score = kmeans_clustering(x_train,10,50,0,0)
#     losses = np.append(losses,loss)
#     CH_mat = np.append(CH_mat,CH_score)


# plt.plot(ratio,losses,c='red')
# plt.xlabel('Sample size ratio')
# plt.ylabel('Loss')
# plt.show()

# plt.plot(ratio,CH_mat,c='magenta')
# plt.xlabel('Sample size ratio')
# plt.ylabel('CH_score')
# plt.show()


# # ### varying number of clusters

# losses = []
# CH_mat = []
# sim_mat = []
# for i in no_of_clusters:
#     labels, loss, CH_score, similarity = kmeans_clustering(X,i,50,0,0)
#     losses = np.append(losses,loss)
#     CH_mat = np.append(CH_mat,CH_score)

# plt.plot(no_of_clusters,losses,c='green')
# plt.xlabel('No. of clusters K')
# plt.ylabel('Loss')
# plt.show()

# plt.plot(no_of_clusters,CH_mat,c='cyan')
# plt.xlabel('No. of clusters K')
# plt.ylabel('CH_score')
# plt.show()


# # ### initializing from random points

# losses = []
# CH_mat = []
# sim_mat = []
# for i in range(0,10):
#     labels, loss, CH_score, similarity = kmeans_clustering(X,10,50,0,1)
#     losses = np.append(losses,loss)
#     CH_mat = np.append(CH_mat,CH_score)

# plt.plot(list(range(1,11)),losses,c='blue')
# plt.ylabel('Loss')
# plt.show()

# plt.plot(list(range(1,11)),CH_mat,c='red')
# plt.ylabel('CH_score')
# plt.show()
