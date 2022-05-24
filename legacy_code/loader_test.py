import torch, os
import torchvision
import wandb
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from utils import hinge_loss
from train import train
from data.confounder_utils import prepare_confounder_data
from data import dro_dataset
from tqdm import tqdm

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

os.putenv("CUDA_VISIBLE_DEVICES", "1")
device = torch.device("cuda")

pretrained = True
n_classes = 2

model_name = "fc"

if model_name == "pretrained-50":
    model = torchvision.models.resnet50(pretrained=pretrained)
elif model_name == "pretrained-18":
    model = torchvision.models.resnet18(pretrained=pretrained)
elif model_name == "fc":
    model_dir = "results/CUB/CUB_resnet_fc_grads/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs/"
    model = torch.load(model_dir + "160_model.pth")
    model.eval()
elif model_name == "fcfc":
    model_dir = "results/CUB/CUB_resnet_fc_fc_nodrop_grads/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs/"
    model = torch.load(model_dir + "160_model.pth")
    model.eval()

model = model.to(device)

loss_fn = nn.CrossEntropyLoss(reduction="none")

class DefaultArgs:
    def __init__(self):
        self.root_dir = "./cub"
        self.dataset = "CUB"
        self.target_name = "waterbird_complete95"
        self.confounder_names = ["forest2water2"]
        self.model = "resnet50"
        self.metadata_csv_name = None
        self.augment_data = False
        self.fraction = 1.0
        self.override_groups_file=False

def get_data(which="train", get_grads=False):
    loader_kwargs = {"batch_size": 64,
                     "num_workers": 4,
                     "pin_memory": True}
    args = DefaultArgs()

    train_data, val_data, test_data = prepare_confounder_data(args, train=True, return_full_dataset=False)
    all_data = prepare_confounder_data(args, train=False, return_full_dataset=True)

    train_loader = dro_dataset.get_loader(train_data,
                                          train=True,
                                          reweight_groups=None,
                                          **loader_kwargs)

    val_loader = dro_dataset.get_loader(val_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)
    
    test_loader = dro_dataset.get_loader(test_data,
                                         train=False,
                                         reweight_groups=None,
                                         **loader_kwargs)

    all_loader = dro_dataset.get_loader(all_data,
                                        train=False,
                                        reweight_groups=None,
                                        **loader_kwargs)

    if which=="train":
        loader = train_loader
    elif which=="val":
        loader = val_loader
    elif which=="test":
        loader = test_loader
    else: loader = all_loader
    
    all_embed = []
    all_y = []
    all_g = []
    all_l = []
    all_idx = []
    gs = []
    
    for idx, batch in enumerate(tqdm(loader)):
        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        l = batch[3]
        data_idx = batch[4]

        outputs = model(x)

        if idx == 0:
            all_embed = full_detach(outputs)
            all_y = full_detach(y)
            all_g = full_detach(g)
            all_l = full_detach(l)
            all_idx = full_detach(data_idx)
        else:
            all_embed = np.concatenate([all_embed, full_detach(outputs)], axis=0)
            all_y = np.concatenate([all_y, full_detach(y)])
            all_g = np.concatenate([all_g, full_detach(g)])
            all_l = np.concatenate([all_l, full_detach(l)])
            all_idx = np.concatenate([all_idx, full_detach(data_idx)])

        if get_grads:
            with torch.set_grad_enabled(True):
                for i, pt in enumerate(x):
                    ptpred = torch.squeeze(model(pt[None, :]), dim=1)
                    loss = loss_fn(ptpred, y[i:i+1])
                    loss.backward()
                    
                    if model_name == "fc":
                        last_g = full_detach(model.fc.weight.grad)
                        last_g_bias = full_detach(model.fc.bias.grad)
                    elif model_name == "fcfc":
                        hidden_g = full_detach(model.fc.fc1.weight.grad)
                        hidden_g_bias = full_detach(model.fc.fc1.bias.grad)
                        last_g = full_detach(model.fc.fc2.weight.grad)
                        last_g_bias = full_detach(model.fc.fc2.bias.grad)

                    # subtract one from other for last fc layer grads
                    last_g = last_g[0,:]-last_g[1,:]
                    last_g_bias = last_g_bias[0]-last_g_bias[1]
                    last_g = np.append(last_g, last_g_bias)
                    #print("last layer grad shape", last_g.shape)
                    
                    if model_name == "fcfc":
                        # flatten hidden layer grads
                        hidden_g = np.append(hidden_g, np.reshape(hidden_g_bias, (hidden_g_bias.size, -1)), axis=1)
                        hidden_g = hidden_g.flatten()
                        #print("hidden layer grad shape", hidden_g.shape)

                        # append two layers gradients together
                        last_g = np.append(hidden_g, last_g)
                        #print("final grad shape", last_g.shape)

                    gs.append(last_g)
                    with torch.no_grad():
                        model.zero_grad()
    
    if get_grads:
        gs = np.stack(gs, axis=0)
        print("Gradients have shape", gs.shape)
        
    print(all_embed.shape)
    print(all_y.shape)
    print(all_g.shape)
    return gs, all_y, all_g, all_l, all_idx

should_grads = True
gs1, y1, g1, l1, i1 = get_data("train", get_grads=should_grads)
gs2, y2, g2, l2, i2 = get_data("val", get_grads=should_grads)
gs3, y3, g3, l3, i3 = get_data("test", get_grads=should_grads)
gs5 = np.vstack([gs1, gs2, gs3])
y5 = np.concatenate([y1, y2, y3])
g5 = np.concatenate([g1, g2, g3])
l5 = np.concatenate([l1, l2, l3])
i5 = np.concatenate([i1, i2, i3])

y5 = np.array([x for _, x in sorted(zip(i5, y5))])
g5 = np.array([x for _, x in sorted(zip(i5, g5))])
l5 = np.array([x for _, x in sorted(zip(i5, l5))])
gs5 = np.array([x for _, x in sorted(zip(i5, gs5))])

gs4, y4, g4, l4, i4 = get_data("all", get_grads=should_grads)
y4 = np.array([x for _, x in sorted(zip(i4, y4))])
g4 = np.array([x for _, x in sorted(zip(i4, g4))])
l4 = np.array([x for _, x in sorted(zip(i4, l4))])
gs4 = np.array([x for _, x in sorted(zip(i4, gs4))])

print(np.array_equal(y5, y4))
print(np.array_equal(g5, g4))
print(np.array_equal(l5, l4))
print(np.array_equal(gs5, gs4))
