import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from spherical_kmeans import spherical_kmeans

do_dbscan = False
compute_dists = do_dbscan or True 
do_agg = False
do_kmeans = True
spherekmeans = False and do_kmeans
pca_setting = "scree" # "bias_separate" OR "scree" OR "3D"
dim_red = "PCA"
dist_metric = "cosine" #"cosine" #"euclidean"
overwrite = True
which_data = "val"

data_dir = "../weight_bias_fc_grads_"+which_data+".npy"
grads = np.load(data_dir)
train_l = np.load("../" + which_data +"_data_l_resnet_fc.npy")
print("Loaded gradients of shape", grads.shape)
print(grads.shape[0], "data points,", grads.shape[1], "gradients")

def make_gif(data, color, title, fname, elev=30, granularity=6):
    print("Generating images from different angles.")
    for angle in tqdm(range(0, 360, granularity)):
        fig = plt.figure()
        ax = Axes3D(fig)
        scattered = ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap="Spectral")
        ax.view_init(elev, angle)

        ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
        legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")
        plt.savefig("3dplotfiles/" + fname + "_" + str(angle) + ".png")
        plt.close(fig)       


if compute_dists:
    distance_matrix = pdist(grads, metric=dist_metric)
    avg_distance = distance_matrix.mean()
    print("Avg pairwise " + dist_metric + " distance:", avg_distance)
    fig = plt.figure()
    plt.hist(distance_matrix, bins='auto')
    plt.title("Histogram of pairwise " +dist_metric+" distances, "+which_data)
    plt.savefig("histogram_dists_"+dist_metric+"_"+which_data+".pdf")

    fig = plt.figure()
    plt.plot(list(range(distance_matrix.size)), np.sort(distance_matrix, axis=0))
    plt.title("k-distance plot of pairwise " + dist_metric + "_" + which_data + ".pdf")
    plt.savefig("kdistance_"+dist_metric+"_"+which_data+".pdf")

    def square_to_condensed(i, j, n):
        if i==j: return 0
        if i<j:
            i, j = j, i
        return n*j - j*(j+1)//2 + i - 1 - j

    # compute average/histogram within and cross group distances
    sums_groups = np.zeros((5,5))
    counts_groups = np.zeros((5,5))
    nums_groups = {i:{j:[] for j in range(5)} for i in range(5)}
    n = grads.shape[0]
    for a in tqdm(range(grads.shape[0])):
        for b in range(a+1, grads.shape[0]):
            nums_groups[train_l[a]][train_l[b]].append(distance_matrix[square_to_condensed(a, b, n)])
            nums_groups[train_l[b]][train_l[a]].append(distance_matrix[square_to_condensed(a, b, n)])
            sums_groups[train_l[a], train_l[b]] += distance_matrix[square_to_condensed(a, b, n)]
            sums_groups[train_l[b], train_l[a]] += distance_matrix[square_to_condensed(a, b, n)]
            counts_groups[train_l[a], train_l[b]] += 1
            counts_groups[train_l[b], train_l[a]] += 1
    avgs_groups = sums_groups/counts_groups
    print("Avg group distances")
    print(avgs_groups)
    fig = plt.figure()
    ax = sns.heatmap(avgs_groups)
    plt.title("Average between group " +dist_metric+" distances, "+which_data)
    plt.savefig(dist_metric+"_group_distances_heatmap_"+which_data+".pdf")

    print("Plotting matrix of histograms...")
    fig = plt.figure()
    c = 1
    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(5, 5, c)
            ax.set_xlim(0.25, 2)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.hist(nums_groups[i][j], bins='auto')
            c+=1
    fig.tight_layout()
    plt.savefig(dist_metric + "_cross_group_distances_"+which_data+".pdf")

if dim_red == "MDS":
    print("Dimensionality reduction via MDS")
    if not os.path.exists("mds_data_"+which_data+".npy"):
        m = dist_metric if dist_metric=="euclidean" else "precomputed"
        input_d = grads if dist_metric=="euclidean" else squareform(distance_matrix)
        plotdata = MDS(n_components=3, dissimilarity=m).fit_transform(input_d)
        np.save("mds_data_"+which_data+".npy", plotdata)
    else:
        plotdata = np.load("mds_data_"+which_data+".npy")
elif dim_red == "PCA":
    print("Dimensionality reduction via PCA")
    if not os.path.exists("pca_data_"+which_data+".npy") or overwrite:
        if pca_setting=="bias_separate":
            plotdata = PCA(n_components=2).fit_transform(grads[:,:-1])
            plotdata = np.append(plotdata, np.array([grads[:,-1]]).T, axis=1)
        elif pca_setting=="scree":
            pca = PCA(n_components=60)
            plotdata = pca.fit_transform(grads)
            fig = plt.figure()
            sns.lineplot(np.arange(pca.n_components_) + 1, pca.explained_variance_ratio_, marker="o")
            plt.title("Scree plot for PCs of gradients, "+which_data)
            plt.xlabel("Principal component")
            plt.ylabel("Explained variance")
            plt.savefig("scree_plot_"+which_data+".pdf")
        elif pca_setting=="3D":
            plotdata = PCA(n_components=3).fit_transform(grads)
        np.save("pca_data_"+which_data+".npy", plotdata)
    else:
        plotdata = np.load("pca_data_"+which_data+".npy")

num_pcs = -1
if do_dbscan:
    eps_options = [avg_distance*i/100 for i in range(10, 500, 20)]
    ms_options = list(range(1, 100, 5))
    for ep in eps_options:
        for ms in ms_options:
            dbscan = cluster.DBSCAN(eps=ep, min_samples=ms, metric=dist_metric)
            clustered = dbscan.fit_predict(plotdata[:,:num_pcs]) #grads
            num_clusters = np.unique(clustered).size-1
            print("eps={} ms={} yielded {} clusters".format(ep, ms, num_clusters))
            print(Counter(clustered))

            if num_clusters > 2:
                fig = plt.figure()
                ax = Axes3D(fig)
                scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=clustered, cmap="Spectral")
                ax.text2D(0.05, 0.95, str(num_clusters) + " clusters + outliers, "+which_data+", "+str(num_pcs)+" PCs", transform=ax.transAxes)
                legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")
                plt.savefig("cluster_" + which_data + "_ep_" + str(ep)[:5] + "_ms_" + str(ms) + ".pdf")

if do_kmeans:
    # plot ground truth 
    fig = plt.figure()
    ax = Axes3D(fig)
    scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=train_l, cmap="Spectral")
    ax.text2D(0.05, 0.95, "ground truth groups "+which_data, transform=ax.transAxes)
    legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Groups")
    plt.savefig("groundtruth_groups_" + which_data + ".pdf")
    
    if spherekmeans:
        norm_data = preprocessing.normalize(grads) #plotdata[:,:num_pcs])
        km = spherical_kmeans(norm_data, k=4, s_iters=5)
    else:
        km = cluster.KMeans(n_clusters=4).fit_predict(plotdata[:,:num_pcs])
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
    np.save("../cub/data/waterbird_complete95_forest2water2/cluster_memberships.npy", km)
    fig = plt.figure()
    ax = Axes3D(fig)
    scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=km, cmap="Spectral")
    ax.text2D(0.05, 0.95, "kmeans clusters, "+which_data+", "+str(num_pcs)+" PCs", transform=ax.transAxes)
    legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")
    plt.savefig("cluster_kmeans_" + which_data + ".pdf")

if do_agg:
    agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=175)
    #agg = cluster.AgglomerativeClustering(n_clusters=4)
    labels = agg.fit_predict(grads)
    print(Counter(labels))
    print(agg.n_clusters_)
    print(confusion_matrix(train_l, labels))

# read in true labels
'''train_y = np.load("../train_data_y_resnet50.npy")
train_g = np.load("../train_data_g_resnet50.npy")
print(train_g)
# y: 0 is land, 1 is water
# g: 0 is land, 1 is water

label = []
for i in range(len(train_y)):
    if train_y[i]==0:
        if train_g[i]==0:
            label.append(0)
        else: label.append(1)
    else:
        if train_g[i]==0:
            label.append(2)
        else: label.append(3)
'''

title = dim_red+": 4 groups + outliers, "+which_data
fname = dim_red+"_ground_truth_memberships_"+which_data
make_gif(plotdata, train_l, title, fname)

'''
fig = plt.figure()
ax = Axes3D(fig)
scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=train_l, cmap="Spectral")
ax.text2D(0.05, 0.95, dim_red+": 4 groups + outliers, "+which_data, transform=ax.transAxes)
legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Groups")
ax.add_artist(legend)
plt.savefig(dim_red+"_ground_truth_memberships_"+which_data+".pdf")'''
