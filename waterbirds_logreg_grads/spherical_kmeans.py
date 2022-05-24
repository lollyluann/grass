import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn import preprocessing
from sklearn.decomposition import PCA

def calculate_cost(X, centroids, cluster):
    sumv = 0
    for i, val in enumerate(X):
        sumv += np.sqrt((centroids[int(cluster[i]), 0] - val[0])**2 + (centroids[int(cluster[i]), 1]-val[1])**2)
    return sumv

def spherical_kmeans(X, k, s_iters=5):
    diff = 1
    sphere_it = s_iters
    cluster = np.zeros(X.shape[0])
    centroids = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[centroids, :]

    while sphere_it > 0:
        # for each obs
        for i, row in enumerate(X):
            mn_dist = float('inf')
            # distance of point from all centroids
            for ix, centroid in enumerate(centroids):
                d = np.sum(np.square(centroid-row))
                # get closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = ix
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        if diff==0:
            new_centroids = preprocessing.normalize(new_centroids)
            sphere_it -= 1
        # if centroids are same then exit
        if np.count_nonzero(centroids-new_centroids) == 0:
            print("Switching to centroid normalization")
            diff=0
        centroids = new_centroids
    return centroids, cluster


which_data = "train"

data_dir = "../weight_bias_fc_grads_"+which_data+".npy"
grads = np.load(data_dir)
plotdata = PCA(n_components=5).fit_transform(grads)
train_l = np.load("../" + which_data +"_data_l_resnet50.npy")
print("Loaded gradients of shape", grads.shape)
print(grads.shape[0], "data points,", grads.shape[1], "gradients")

norm_data = preprocessing.normalize(grads) #plotdata[:,:num_pcs])
cents, km = spherical_kmeans(norm_data, k=4, s_iters=5)
print(cents)

c_counts = Counter(km)
print("Kmeans results: ", c_counts)
num_real_clusters = sum([1 for i in c_counts if c_counts[i]>30])
print(num_real_clusters, "clusters with >30 points")

km = np.array([i if c_counts[i]>30 else -1 for i in km])
kmap = {}
new_km = []
j = 0
for i in km:
    if i==-1:
        new_km.append(-1)
    else:
        if i in kmap:
            new_km.append(kmap[i])
        else:
            new_km.append(j)
            kmap[i] = j
            j += 1
km = np.array(new_km)
print(Counter(km))
#np.save("../cub/data/waterbird_complete95_forest2water2/cluster_memberships.npy", km)

fig = plt.figure()
ax = Axes3D(fig)
scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=km, cmap="Spectral")
ax.text2D(0.05, 0.95, "spherical kmeans clusters, "+which_data, transform=ax.transAxes)
legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")
plt.savefig("cluster_kmeans_sphere_" + which_data + ".pdf")
