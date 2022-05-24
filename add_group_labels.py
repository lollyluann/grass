import os
import pandas as pd

data_dir = "cub/data/waterbird_complete95_forest2water2"

metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
new_labels = []
y_c = metadata_df['y'].values
g_c = metadata_df['place'].values

for i in range(len(metadata_df.index)):
    if y_c[i]==0:
        if g_c[i]==0:
            new_labels.append(0)
        else: new_labels.append(1)
    else:
        if g_c[i]==0:
            new_labels.append(2)
        else: new_labels.append(3)

metadata_df["group_labels"] = new_labels
metadata_df.to_csv(os.path.join(data_dir, "metadata.csv"), index=False)
