import torch, math, os
from tqdm import tqdm
import numpy as np
from utils import simul_x_y_a, add_outliers, plot_sample, plot_decision, plot_grad, plot_3d
from metrics import group_metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sampler import BalancedBatchSampler
import matplotlib.pyplot as plt

os.putenv("CUDA_VISIBLE_DEVICES", "1")
device = torch.device("cuda")

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

modelname = "pretrained-50"
train_x = torch.tensor(np.load("../extracted/train_data_resnet_"+modelname+".npy"), device=device).float()
train_y = torch.tensor(np.load("../extracted/train_data_y_resnet_"+modelname+".npy"), device=device).float()
train_l = np.load("../extracted/train_data_l_resnet_"+modelname+".npy")
train_data = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_data, sampler=BalancedBatchSampler(train_data, train_y), batch_size=64)

val_x = torch.tensor(np.load("../extracted/val_data_resnet_"+modelname+".npy"), device=device).float()
val_y = torch.tensor(np.load("../extracted/val_data_y_resnet_"+modelname+".npy"), device=device).float()
val_l = np.load("../extracted/val_data_l_resnet_"+modelname+".npy")

test_x = torch.tensor(np.load("../extracted/test_data_resnet_"+modelname+".npy"), device=device).float()
test_y = torch.tensor(np.load("../extracted/test_data_y_resnet_"+modelname+".npy"), device=device).float()
test_l = np.load("../extracted/test_data_l_resnet_"+modelname+".npy")

'''
a = train_x.shape[1]
weights = torch.randn(a, 1, device=device, dtype=torch.double)/math.sqrt(a)
weights.requires_grad_()
bias = torch.zeros(1, requires_grad=True, device=device, dtype=torch.double)

def model(xb):
    sig = torch.nn.Sigmoid().to(device)
    return sig(xb @ weights + bias)
'''
def predict(model, data):
    model.eval()
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return full_detach(model(data).round())

def evaluate(model, data, labels, groups, verbose=False):
    if isinstance(labels, torch.Tensor):
        labels = full_detach(labels)
    predictions = predict(model, data)
    acc = accuracy_score(labels, predictions)

    group_accs = [0,0,0,0,0]
    for ig in [0, 1, 2, 3, 4]:
        indices = [i for i in range(len(labels)) if (groups[i]==ig)]
        if not indices: continue
        group_accs[ig] = accuracy_score(labels[indices], predictions[indices])
        if verbose:
            print("Group", ig, ":", len(indices), "samples")
            print(group_accs[ig])
    if verbose:
        print("Overall accuracy", acc)
        print(confusion_matrix(labels, predictions))
    return acc, group_accs

def export_grads(x, y, model, optimizer, data_name, loss_func=torch.nn.BCELoss()):
    weight_traingrad, bias_traingrad = [], []
    for i, pt in enumerate(x):
        ptpred = torch.squeeze(model(pt[None, :]), dim=1)
        loss2 = loss_func(ptpred, y[i:i+1])
        loss2.backward()
        weight_grad = full_detach(model.linear.weight.grad)
        bias_grad = full_detach(model.linear.bias.grad)
        weight_traingrad.append(weight_grad.copy())
        bias_traingrad.append(bias_grad.copy())
        with torch.no_grad():
            model.zero_grad()

    weight_traingrad = np.array(weight_traingrad)
    bias_traingrad = np.array(bias_traingrad)

    grads = np.append(weight_traingrad, bias_traingrad[np.newaxis].T, axis=1)
    #print(weight_traingrad.shape, bias_traingrad.shape, grads.shape)
    save_dir = "../extracted/"+modelname+"_weight_bias_grads_"+data_name+".npy"
    np.save(save_dir, grads)

model = LogisticRegression(train_x.shape[1], 1)
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

weight_traingrad, input_traingrad, bias_traingrad = [], [], []
weight_testgrad, input_testgrad, bias_testgrad = [], [], []

losses = []
accs = []
g_accs = []

model.train()
model.to(device)

e_to_extract = {5, 65}

for epoch in tqdm(range(epochs)):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x).squeeze()
        loss = loss_func(pred, batch_y)
        loss.backward()
        optimizer.step()

    if epoch%5==0:
        with torch.no_grad():
            val_outputs = model(val_x).squeeze()
            val_loss = loss_func(val_outputs, val_y)
            losses.append(val_loss.item())

            val_acc, group_accs = evaluate(model, val_x, val_y, val_l)
            accs.append(val_acc)
            g_accs.append(group_accs)

    if epoch in e_to_extract:
        export_grads(train_x, train_y, model, optimizer, "train_epoch"+str(epoch))
        export_grads(val_x, val_y, model, optimizer, "val_epoch"+str(epoch))
        export_grads(test_x, test_y, model, optimizer, "test_epoch"+str(epoch))
       
        with torch.no_grad():
            print("Training data accuracies")
            evaluate(model, train_x, train_y, train_l, verbose=True)

            print("Validation data accuracies")
            evaluate(model, val_x, val_y, val_l, verbose=True)

            print("Test data accuracies")
            evaluate(model, test_x, test_y, test_l, verbose=True)



yticks = list(range(0, epochs, 5))
plt.plot(yticks, losses)
plt.axvline(x=np.argmin(losses)*5, linestyle="--", linewidth=0.5, c='k')
plt.ylabel("Loss")
plt.title("Loss over time")
plt.savefig("val_loss.pdf")
plt.close()
print("Epoch with best val loss:", np.argmin(losses), min(losses))

g_accs = np.array(g_accs)
plt.plot(yticks, accs, label="Average accuracy")
plt.plot(yticks, g_accs[:, 0], label="Group 0 accuracy")
plt.plot(yticks, g_accs[:, 1], label="Group 1 accuracy")
plt.plot(yticks, g_accs[:, 2], label="Group 2 accuracy")
plt.plot(yticks, g_accs[:, 3], label="Group 3 accuracy")
plt.plot(yticks, g_accs[:, 4], label="Outliers accuracy")
plt.axvline(x=np.argmax(accs)*5, linestyle="--", linewidth=0.5, c='k')
plt.title("Accuracies over time")
plt.legend(prop={'size':6})
plt.savefig("val_accuracies.pdf")
plt.close()
print("Epoch with best val acc:", np.argmax(accs), max(accs))

## Base classifier
#base_predict = predict(model, train_x)
#print('Baseline')
#_ = group_metrics(full_detach(train_y), base_predict, test_a, label_protected=1, label_good=0)
#print('Test biased accuracy', np.mean(predict(model, test_biased_x) == test_biased_y))

## Base ideal classifier
'''base_lr_ideal = LogisticRegression(solver='liblinear', fit_intercept=True)
base_lr_ideal.fit(test_x, test_y)
base_predict_ideal = base_lr_ideal.predict(test_x)
print('\nBaseline IDEAL')
_ = group_metrics(test_y, base_predict_ideal, test_a, label_protected=1, label_good=0)
'''
#plot_decision(full_detach(test_x), test_a, full_detach(test_y), lambda x: predict(model, x), title='Log Reg')
#plot_decision(test_x, test_a, test_y, lambda x: base_lr_ideal.predict_proba(x)[:,1], title='Log Reg IDEAL')

print("Training data accuracies")
evaluate(model, train_x, train_y, train_l, verbose=True)

print("Validation data accuracies")
evaluate(model, val_x, val_y, val_l, verbose=True)

print("Test data accuracies")
evaluate(model, test_x, test_y, test_l, verbose=True)

#export_grads(train_x, train_y, model, optimizer, "train")
#export_grads(val_x, val_y, model, optimizer, "val")
#export_grads(test_x, test_y, model, optimizer, "test")



