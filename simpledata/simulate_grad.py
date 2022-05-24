import torch, math
from tqdm import tqdm
import numpy as np
from utils import simul_x_y_a, add_outliers, plot_sample, plot_decision, plot_grad, plot_3d
from sklearn.linear_model import LogisticRegression
from metrics import group_metrics

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

mu_mult = 2.
cov_mult = 1.
skew = 5.

train_prop_mtx = [[0.4, 0.1],[0.4, 0.1]]
train_x, train_a, train_y = simul_x_y_a(train_prop_mtx, n=1000, mu_mult=mu_mult, 
                                        cov_mult=cov_mult, skew=skew, outliers=True)
plot_sample(train_x, train_a, train_y, title='Train')

test_prop_mtx = [[0.25, 0.25],[0.25, 0.25]]
test_x, test_a, test_y = simul_x_y_a(test_prop_mtx, n=1000, mu_mult=mu_mult, 
                                     cov_mult=cov_mult, skew=skew, outliers=False)
plot_sample(test_x, test_a, test_y, title='Test')

train_x, train_y, text_x, test_y = map(torch.tensor, (train_x, train_y, test_x, test_y))
train_x.requires_grad_()
test_x = torch.tensor(test_x)
test_x.requires_grad_()
train_y.double()

test_biased_x, test_biased_a, test_biased_y = simul_x_y_a(train_prop_mtx, n=1000, 
                                                          mu_mult=mu_mult, cov_mult=cov_mult,
                                                          skew=skew)

a = train_x.shape[1]
weights = torch.randn(a, 1, dtype=torch.double)/math.sqrt(a)
weights.requires_grad_()
bias = torch.zeros(1, requires_grad=True, dtype=torch.double)

def model(xb):
    return torch.nn.Sigmoid()(xb @ weights + bias)

loss_func = torch.nn.BCELoss()
lr = 0.05
epochs = 30

weight_traingrad, input_traingrad, bias_traingrad = [], [], []
weight_testgrad, input_testgrad, bias_testgrad = [], [], []
ugh = []

for epoch in tqdm(range(epochs)):
    pred = model(train_x).squeeze()
    loss = loss_func(pred, train_y.double())
    loss.backward()
    input_traingrad = full_detach(train_x.grad)
    #weight_traingrad = weights.grad.squeeze().detach().cpu().numpy()
    #bias_traingrad = bias.grad.squeeze().detach().cpu().numpy()
   
    with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr
        weights.grad.zero_()
        bias.grad.zero_()
    
    weight_traingrad, bias_traingrad = [], []
    for i, pt in enumerate(train_x):
        ptpred = torch.squeeze(model(pt[None, :]), dim=1)
        loss2 = loss_func(ptpred, train_y[i:i+1].double())
        loss2.backward()
        weight_grad = full_detach(weights.grad)
        bias_grad = full_detach(bias.grad)
        weight_traingrad.append(weight_grad.copy())
        bias_traingrad.append(bias_grad.copy())
        with torch.no_grad():
            weights.grad.zero_()
            bias.grad.zero_()
    
    '''
    testloss = loss_func(model(test_x), test_y)
    testloss.backward()
    input_testgrad = test_x.grad.squeeze().detach().cpu().numpy()
    weight_testgrad = weights.grad.squeeze().detach().cpu().numpy()
    bias_testgrad = bias.grad.squeeze().detach().cpu().numpy()
    weights.grad.zero_()
    bias.grad.zero_()'''


def predict(model, data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return full_detach(model(data).round())

## Base classifier
base_predict = predict(model, test_x)
print('Baseline')
_ = group_metrics(full_detach(test_y), base_predict, test_a, label_protected=1, label_good=0)
print('Test biased accuracy', np.mean(predict(model, test_biased_x) == test_biased_y))

plot_decision(full_detach(test_x), test_a, full_detach(test_y), lambda x: predict(model, x), title='Log Reg')

weight_traingrad = np.array(weight_traingrad)
bias_traingrad = np.array(bias_traingrad)
print(weight_traingrad.shape, input_traingrad.shape, bias_traingrad)
print(weight_traingrad.sum(), bias_traingrad.sum())
plot_grad(full_detach(train_x), train_a, full_detach(train_y), input_traingrad, title="TrainGradInput")
plot_grad(full_detach(train_x), train_a, full_detach(train_y), weight_traingrad, title="TrainGradWeight")
plot_3d(full_detach(train_x), full_detach(train_y), train_a, weight_traingrad, bias_traingrad, title="TrainGradWeightsBias")

grads = np.append(weight_traingrad, bias_traingrad[np.newaxis].T, axis=1)
print(weight_traingrad.shape, bias_traingrad.shape, grads.shape)
save_dir = "weight_bias_grads.npy"
np.save(save_dir, grads)








