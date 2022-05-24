import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
import pandas as pd
import argparse
from cluster_utils import load_class_data, iou_adaptive, internal_evals, cluster_and_extract

x_axis_labels = [5, 10, 20, 30, 40, 50, 60, 100]  # labels for x-axis
y_axis_labels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]  # labels for y-axis

def visualize_internal_evals(chi_mat, dbs_mat, sil_mat):
    # Visualize internal cluster evaluation metrics as heatmaps
    s = sns.heatmap(np.array(chi_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Calinski-Harabasz score for different eps and m")

    sns.heatmap(np.array(dbs_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Davies-Bouldin score for different eps and m")

    sns.heatmap(np.array(sil_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Silhouette score for different eps and m")

    print("You must include lines to save or show the plots in this function to view them.")

def nmi_matrix(clusterings):
    nmi_mat = []

    for i in range(len(clusterings)):
        nmi_row = []
        for j in range(len(clusterings)):
            nmi_row.append(NMI(clusterings[i], clusterings[j]))
        nmi_mat.append(nmi_row)

    nmi_mat = np.array(nmi_mat)
    new_labels = [(i, j) for i in y_axis_labels for j in x_axis_labels]

    plt.figure(figsize=(16, 14))
    s = sns.heatmap(nmi_mat, xticklabels=new_labels, yticklabels=new_labels)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    s.set_title("NMI score between clusterings with different (eps, m)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pretrained-50")
    parser.add_argument("--out_dir", type=str, default="cub/data/waterbird_complete95_forest2water2")
    parser.add_argument("--eps", type=float, default=0.4)
    parser.add_argument("--min_samples", type=int, default=100)

    args = parser.parse_args()
    cluster_and_extract(eps=args.eps, ms=args.min_samples, modelname=args.model_name, out_dir=args.out_dir)


    # DBSCAN testing with true group labels
    '''dist, grads, groups, all_i = load_class_data(classi=0)
    
    arss = []
    nmis = []
    ious = []
    ious2 = []
    
    chi_mat, dbs_mat, sil_mat= [], [], []
    
    clusterings = []
    
    best_mean = 0
    best_params = {}
    for eps in np.linspace(0.1, 0.7, 13)  :# (0.3,0.6,10):
        chi_row, dbs_row, sil_row = [], [], []
        # for min_samples in [3, 5, 10, 15, 20, 30]:
        for min_samples in [5, 10, 20, 30, 40, 50, 60, 100]:
            # cluster using precomputed cosine distance matrix
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            dbscan.fit(dist)
    
            # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            # dbscan.fit(grads)
    
            chi, dbs, sil = internal_evals(grads, dist, dbscan.labels_)
            chi_row.append(chi)
            dbs_row.append(dbs)
            sil_row.append(sil)
            clusterings.append(dbscan.labels_)
    
            print('Eps', eps, 'Min samples', min_samples)
            print('Cluster counts', np.unique(dbscan.labels_, return_counts=True))
    
            ars = ARS(groups, dbscan.labels_)
            arss.append(ars)
            nmi = NMI(groups, dbscan.labels_)
            nmis.append(nmi)
            print('ARS', ars, 'NMI', nmi)
    
            iou, iou2, eq = iou_adaptive(groups, dbscan.labels_)
            ious.append(iou)
            ious2.append(iou2)
    
            if eq:
                val = sum(iou) + sum(iou2) + iou[1] + iou2[0] + (iou[2 ] +iou2[2] ) /2
                val /= 9
                print("Weighted avg", val)
                if val > best_mean:
                    best_mean = val
                    best_params['eps'] = eps
                    best_params['min_samples'] = min_samples
    
            print('\n')
    
        chi_mat.append(chi_row)
        dbs_mat.append(dbs_row)
        sil_mat.append(sil_row)
    
    print(50 *'-')
    print('DBSCAN: best ARS', max(arss), 'best NMI', max(nmis))
    print('DBSCAN: best IOU', np.max(ious, axis=0))
    print('DBSCAN: best IOU 2', np.max(ious2, axis=0))
    
    print("DBSCAN: best avg IOU", best_mean)
    print("best avg IOU params", best_params)
    
    visualize_internal_evals(chi_mat, dbs_mat, sil_mat)
    nmi_matrix(clusterings)
    '''
