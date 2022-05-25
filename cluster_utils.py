import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import adjusted_rand_score as ARS
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pandas as pd


def cosine_dist(X1, X2):
    dist = 1 - X1 @ X2.T
    return np.maximum(dist, 0.)


def e_dist(A, B, cosine=False, eps=1e-10):
    ## I don't use this function - it is for pairwise Euclidean mostly

    A_n = (A ** 2).sum(axis=1).reshape(-1, 1)
    B_n = (B ** 2).sum(axis=1).reshape(1, -1)
    inner = A @ B.T
    if cosine:
        return 1 - inner / (np.sqrt(A_n * B_n) + eps)
    else:
        return np.maximum(0., A_n - 2 * inner + B_n)


def get_dist_hist(X1, X2):
    dist = cosine_dist(X1, X2)
    # dist = e_dist(X1, X2)

    n, m = dist.shape
    if n > m:
        dist = dist.T
        n, m = m, n
    if np.allclose(np.diag(dist), 0):
        k = 1
    else:
        k = 0

    dist = dist[np.triu_indices(n, k, m)]

    return dist


def iou_compute(idx1, idx2):
    # proportion of true points that are in the predicted group
    inter = len(np.intersect1d(idx1, idx2))
    union = len(np.union1d(idx1, idx2))
    # return inter/union

    # proportion of predicted points that are in the true group
    # proportion of true points that are in the predicted group
    return inter / len(idx1), inter / len(idx2) if len(idx2) != 0 else 0


def iou_stat(label_true, label_pred):
    # Outliers
    out_true, out_pred = np.where(label_true == 2)[0], np.where(label_pred == -1)[0]
    out_iou, _ = iou_compute(out_true, out_pred)

    # Majority
    maj_true, maj_pred = np.where(label_true == 0)[0], np.where(label_pred == 0)[0]
    maj_iou, _ = iou_compute(maj_true, maj_pred)

    ## Minority
    min_true, min_pred = np.where(label_true == 1)[0], np.where(label_pred > 0)[0]
    min_iou, _ = iou_compute(min_true, min_pred)
    return [out_iou, maj_iou, min_iou]


def iou_adaptive(label_true, label_pred, return_map=False):
    best_avg_iou = np.array([0, 0, 0])
    best_avg_iou2 = np.array([0, 0, 0])
    best_avg_inds = {}
    best_avg_inds2 = {}

    # true_perms = itertools.permutations([0, 1, 2])
    # for i in true_perms:
    i = [0, 1, 2]
    pred_perms = itertools.permutations([-1, 0, None])
    for j in pred_perms:
        # Majority
        if j[0] == None:
            j0 = np.where(np.logical_and(label_pred != j[2], label_pred != j[1]))
        else:
            j0 = np.where(label_pred == j[0])
        maj_true, maj_pred = np.where(label_true == i[0])[0], j0[0]
        maj_iou, maj_iou2 = iou_compute(maj_true, maj_pred)

        # Minority
        if j[1] == None:
            j1 = np.where(np.logical_and(label_pred != j[0], label_pred != j[2]))
        else:
            j1 = np.where(label_pred == j[1])
        min_true, min_pred = np.where(label_true == i[1])[0], j1[0]
        min_iou, min_iou2 = iou_compute(min_true, min_pred)

        ## Outliers
        if j[2] == None:
            j2 = np.where(np.logical_and(label_pred != j[0], label_pred != j[1]))
        else:
            j2 = np.where(label_pred == j[2])
        out_true, out_pred = np.where(label_true == i[2])[0], j2[0]
        out_iou, out_iou2 = iou_compute(out_true, out_pred)

        cur_iou = np.array([maj_iou, min_iou, out_iou])
        cur_mean = np.mean(cur_iou)
        cur_iou2 = np.array([maj_iou2, min_iou2, out_iou2])
        cur_mean2 = np.mean(cur_iou2)

        if cur_mean > np.mean(best_avg_iou):
            best_avg_iou = cur_iou
            best_avg_inds[i[0]] = j[0] if j[0] != None else "all others"
            best_avg_inds[i[1]] = j[1] if j[1] != None else "all others"
            best_avg_inds[i[2]] = j[2] if j[2] != None else "all others"

        if cur_mean2 > np.mean(best_avg_iou2):
            best_avg_iou2 = cur_iou2
            best_avg_inds2[i[0]] = j[0] if j[0] != None else "all others"
            best_avg_inds2[i[1]] = j[1] if j[1] != None else "all others"
            best_avg_inds2[i[2]] = j[2] if j[2] != None else "all others"

    print("Best avg IOU", best_avg_iou.tolist())
    print(best_avg_inds)

    print("Best avg IOU 2", best_avg_iou2.tolist())
    print(best_avg_inds2)

    if not return_map:
        return best_avg_iou, best_avg_iou2, best_avg_inds == best_avg_inds2
    else:
        return best_avg_iou, best_avg_iou2, best_avg_inds == best_avg_inds2, best_avg_inds, best_avg_inds2


def internal_evals(X, distance_mat, labels):
  chi = calinski_harabasz_score(X, labels)
  dbs = davies_bouldin_score(X, labels)
  sil = silhouette_score(distance_mat, labels, metric="precomputed")
  return chi, dbs, sil


def plot_data(data, labels, title=None):
  fig = plt.figure()
  ax = Axes3D(fig)
  if data.shape[1] != 3:
    data = PCA(n_components=3).fit_transform(data)
  scattered = ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap="viridis") #Spectral")
  legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")

  if title:
    ax.set_title(title)
  plt.show()


def plot_pairwise_distances(grads, groups):
    ## Pairwise distances
    # min_v, max_v = dist.min(), dist.max()
    n_groups = len(np.unique(groups))
    min_v, max_v = 0., 4.  # 2.
    bins = np.linspace(min_v, max_v, 100)
    fig, axes = plt.subplots(nrows=n_groups, ncols=n_groups, figsize=(3 * n_groups, 3 * n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            dist_ij = get_dist_hist(grads[groups == i], grads[groups == j])
            axes[i, j].hist(dist_ij, bins, density=1)
    plt.show()


def cluster_metrics(labels, groups):
  print('Cluster counts', Counter(labels))
  ars = ARS(groups, labels)
  nmi = NMI(groups, labels)
  print('K-Means: ARS', ars, 'NMI', nmi)

  iou_adaptive(groups, labels)

  min_true = np.where(groups==1)[0]
  for i in np.unique(labels):
      print(i, 'Minority IOU:', iou_compute(min_true, np.where(labels==i)[0]))


def load_class_data(classi, epoch, base_folder):
  data_subset = "train"
  modelname = "pretrained-50"
  grads = np.load(base_folder + modelname + '_weight_bias_grads_' + data_subset + '_epoch' + str(epoch) + '.npy')
  groups = np.load(base_folder + data_subset + '_data_l_resnet_'+modelname+'.npy')
  y = np.load(base_folder + data_subset + '_data_y_resnet_'+modelname+'.npy')
  train_i = np.load(base_folder + data_subset + '_data_i_resnet_'+modelname+'.npy')

  data_subset = "val"
  vgrads = np.load(base_folder + modelname + '_weight_bias_grads_' + data_subset + '_epoch' + str(epoch) + '.npy')
  vgroups = np.load(base_folder + data_subset + '_data_l_resnet_'+modelname+'.npy')
  vy = np.load(base_folder + data_subset + '_data_y_resnet_'+modelname+'.npy')
  vi = np.load(base_folder + data_subset + '_data_i_resnet_'+modelname+'.npy')

  # combine train and val data
  grads = np.concatenate([grads, vgrads], axis=0)
  groups = np.concatenate([groups, vgroups], axis=0)
  y = np.concatenate([y, vy], axis=0)
  all_i = np.concatenate([train_i, vi], axis=0)

  class_idx = classi

  groups = groups[y==class_idx]
  grads = grads[y==class_idx]
  all_i = all_i[y==class_idx]

  for i, g in enumerate(np.unique(groups)):
      groups[groups==g] = i

  center = grads.mean(axis=0)

  grads = grads - center
  grads = normalize(grads)

  dist = cosine_dist(grads, grads)
  return dist, grads, groups, all_i


def cluster_and_extract(eps, ms, modelname, epoch, in_dir, out_dir):
    dfc = pd.DataFrame({'idx': np.load(in_dir + 'test_data_i_resnet_' + modelname + '.npy'),
                        'clustered_idx': np.load(in_dir + 'test_data_l_resnet_' + modelname + '.npy')})

    dfs = []
    for j in [0, 1]:
        print("Class {}, eps={}, min_samples={}".format(j, eps, ms))
        dist, grads, _, all_is = load_class_data(classi=j, epoch=epoch, base_folder=in_dir)
        dbscan = DBSCAN(eps=eps, min_samples=ms, metric='precomputed')
        dbscan.fit(dist)
        print("Cluster counts:", Counter(dbscan.labels_))
        plot_data(grads, dbscan.labels_, "Class {} Clustered (eps={}, m={}), train+val".format(j, eps, ms))

        adjusted_labels = dbscan.labels_
        adjusted_labels = [j * 2 + a if a != -1 else -1 for a in adjusted_labels]
        output_labels = pd.DataFrame({'idx': all_is, 'clustered_idx': adjusted_labels})
        dfs.append(output_labels)

    all_output_labels = dfs[0].append(dfs[1], ignore_index=True).append(dfc, ignore_index=True)

    all_output_labels = all_output_labels.sort_values(by=['idx'])
    all_output_labels.idx = all_output_labels.idx + 1
    print("Group counts:", all_output_labels["clustered_idx"].value_counts())

    out_name = out_dir + "train_val_test_labels_" + str(eps) + "_" + str(ms) + ".csv"
    all_output_labels.to_csv(out_name, index=False)
    print("Estimated group labels written to " + out_name)
