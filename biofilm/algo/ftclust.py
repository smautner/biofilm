'''
# cluster features  / select highly correlating ones from each cluster...



- we want to make a umap projection of the features

- draw and color by correlation
- or make a clustermap if the numft is low..

- cluster with gmm and select numclust based on something

- is there a way to ballance uniqueness and correlation?
'''

from lmz import Map, Range
from biofilm import util
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from umap import UMAP
import seaborn as sns
from biofilm.algo import forest
import basics as ba
def ft(x,y):

    score, _ = forest.forest(x,y,class_weight = 'balanced')
    x = select(x,score,10000)
    corr  = spearman(x,y)

    xt = np.transpose(util.zehidense(x))
    xt = xt[~np.all(xt == 0, axis=1).A1]
    print(f"{ xt.shape=}")

    xt = np.abs(np.corrcoef(xt))

    xt = UMAP(n_components=10).fit_transform(xt)

    '''
    draw some projections of the data... PCA UMAP  and a heatmap
    '''
    if False:
        xt2d = PCA(n_components=2).fit_transform(xt)
        plt.scatter(xt2d[:,0],xt2d[:,1],c=corr)
        plt.show(); plt.close()
        xt2d = UMAP(n_components=2,n_neighbors=int(np.sqrt(x.shape[1]))).fit_transform(xt)
        plt.scatter(xt2d[:,0],xt2d[:,1],c=corr)
        plt.show(); plt.close()
        sns.clustermap(xt)
        plt.show(); plt.close()
        plt.plot(np.sort(corr))
        plt.show(); plt.close()


    print("start clustering..")
    print(f"{ xt.shape=}")
    nc = Range(2,int(np.sqrt(xt.shape[0])),3)
    nc = Range(2,100,5)
    #clusterings = [GaussianMixture(n_components=i).fit_predict(xt) for i in nc ]
    clusterings = ba.mpmap(cluster,[(n,xt) for n in nc], poolsize = 20)
    print(f"{ xt.shape=}")

    # each nc gets the average of the maxcorrPerCluster
    def avgmaxcorr(labels):
        topscr = lambda l: np.max(corr[labels == l])
        return np.mean(Map(topscr, np.unique(labels)))
    avgMaxCorPerClust = Map(avgmaxcorr,clusterings)

    plt.scatter(nc, avgMaxCorPerClust)
    plt.show(); plt.close()
    exit()


def cluster(z):
    i,xt = z
    return GaussianMixture(n_components=i).fit_predict(xt)

def select(X,scores,num):
    return X[:,np.argsort(scores)[-num:]]



def spearman(x,y):
    spear = lambda ft: np.abs(spearmanr(ft.T,y)[0])
    x = util.zehidense(x)
    re = Map(spear,x.T)
    return np.array(re)



