from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

train_x = np.load("../train_data_resnet50.npy")
train_y = np.load("../train_data_y_resnet50.npy")
train_g = np.load("../train_data_g_resnet50.npy")

val_x = np.load("../val_data_resnet50.npy")
val_y = np.load("../val_data_y_resnet50.npy")
val_g = np.load("../val_data_g_resnet50.npy")

test_x = np.load("../test_data_resnet50.npy")
test_y = np.load("../test_data_y_resnet50.npy")
test_g = np.load("../test_data_g_resnet50.npy")

def get_group_accs(pred, groups):
    n_groups = np.unique(groups).size
    for i in range(n_groups):
        correct = i//2
        acc = accuracy_score(pred[groups==i], np.full(pred[groups==i].size, correct))
        print("Group", i, "accuracy:", acc)

lr = LogisticRegression(max_iter=500)
lr2 = lr.fit(train_x, train_y)

val_out = lr2.predict(val_x)
print("Validation avg acc:", lr2.score(val_x, val_y))
get_group_accs(val_out, val_g)

test_out = lr2.predict(test_x)
print("Test avg acc:", lr2.score(test_x, test_y))
get_group_accs(test_out, test_g)
