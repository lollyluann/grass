import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from data.confounder_utils import prepare_confounder_data
from data import dro_dataset

from legacy_code.extractable_resnet import resnet18

model_name = "pretrained-18"

if model_name == "pretrained-50":
    model = torchvision.models.resnet50(pretrained=pretrained)
    model = torch.nn.Sequential(*list(model.children())[:-1])
elif model_name == "pretrained-18":
    model = torchvision.models.resnet18(pretrained=pretrained)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = resnet18(pretrained=True, layers_to_extract=1)
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
        self.batch_size = 64
    
def get_data(which="train", get_grads = False):
    loader_kwargs = {"batch_size": 64,
                     "num_workers": 4,
                     "pin_memory": True}
    args = DefaultArgs()

    train_data, val_data, test_data = prepare_confounder_data(args, train=True, return_full_dataset=False)

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

    if which=="train":
        loader = train_loader
        data_set = train_data
    elif which=="val":
        loader = val_loader
        data_set = val_data
    else:
        loader = test_loader
        data_set = test_data

    # Initialize model
    #model = resnet18(
    #    pretrained=True,
    #    layers_to_extract=1)
    #model = torchvision.models.resnet18(pretrained=True)
    #model = torch.nn.Sequential(*list(model.children())[:-1])

    model.eval()
    model = model.cuda()

    n = len(data_set)
    idx_check = np.empty(n)
    last_batch = False
    start_pos = 0
    all_y, all_g = [], []
    gs = []

    with torch.set_grad_enabled(False):
        for i, (x_batch, y, g, l, j) in enumerate(tqdm(loader)):
            x_batch = x_batch.cuda()
            y = y.cuda()
            
            num_in_batch = list(x_batch.shape)[0]

            #end_pos = start_pos + num_in_batch

            all_y.extend(y)
            all_g.extend(g)

            features_batch = model(x_batch).data.cpu().numpy().squeeze()
            if i == 0:
                d = features_batch.shape[1]
                print(f'Extracting {d} features per example')
                #features = np.empty((n, d))
    
                all_embed = features_batch
            else:
                all_embed = np.concatenate([all_embed, features_batch], axis=0)

            #features[start_pos:end_pos, :] = features_batch

            #start_pos = end_pos

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if get_grads:
                with torch.set_grad_enabled(True):
                    for i, pt in enumerate(x_batch):
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
            np.save("weight_bias_"+model_name+"_grads_" + which + ".npy", gs)

    print(all_embed.shape)
    print(np.array(all_y).shape)
    features_mine_path = f'{which}_data_resnet_pretrained-18.npy'
    np.save(features_mine_path, all_embed)
    output_path = f'resnet-18_1layer{which}.npy'
    np.save(output_path, features)
    label_output_path = f'data_{which}_y.npy'
    np.save(label_output_path, all_y)
    group_output_path = f'data_{which}_g.npy'
    np.save(group_output_path, all_g)

def main():
    torch.manual_seed(0)

    get_data("train")
    get_data("val")
    get_data("test")

if __name__=='__main__':
    main()
