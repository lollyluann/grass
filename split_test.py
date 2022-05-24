import pandas as pd
import os
import numpy as np

'''data_dir = "results/CUB/CUB_resnet_grads_test/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs"
grads = np.load(data_dir + "/last_fc_grads_epoch_50.npy")
print(grads.shape)
print(grads)
'''
ddir = "train_data_l_resnet_fc.npy"
#print(np.unique(np.load(ddir)))
#print(np.load(ddir))


data_dir = "cub/data/waterbird_complete95_forest2water2"
out_dir = "cub/data/waterbird_outliers"
outliers_file = "cub/data/outliers_annotations.csv"

metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
metadata_2 = metadata_df[metadata_df.group_labels != 4]

print(metadata_df.split.unique())
print(metadata_df.split.value_counts())
print(metadata_2.split.value_counts())
print(metadata_df.group_labels.value_counts())
print(metadata_df[(metadata_df['split']==0)].group_labels.value_counts())
print(metadata_df[(metadata_df['split']==1)].group_labels.value_counts())
print(metadata_df[(metadata_df['split']==2)].group_labels.value_counts())

rows = metadata_df[(metadata_df['split']==2) & (metadata_df['group_labels']==4)]
print(rows)

