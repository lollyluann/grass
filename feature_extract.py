import torch, os
import torchvision
import torch.nn as nn
import numpy as np

from data.confounder_utils import prepare_confounder_data
from data import dro_dataset
from tqdm import tqdm
import argparse


def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

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


def get_data(nargs, which="train", get_grads=False):
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

    if which == "train":
        loader = train_loader
        dataset = train_data
    elif which == "val":
        loader = val_loader
        dataset = val_data
    else:
        loader = test_loader
        dataset = test_data

    all_embed = []
    all_y = []
    all_g = []
    all_l = []
    all_i = []
    gs = []

    n = len(dataset)

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
        else:
            all_embed = np.concatenate([all_embed, full_detach(outputs)], axis=0)
        all_y.extend(full_detach(y))
        all_g.extend(full_detach(g))
        all_l.extend(full_detach(l))
        all_i.extend(full_detach(data_idx))

        if get_grads:
            with torch.set_grad_enabled(True):
                for i, pt in enumerate(x):
                    ptpred = torch.squeeze(model(pt[None, :]), dim=1)
                    loss = loss_fn(ptpred, y[i:i + 1])
                    loss.backward()

                    if nargs.model_name == "fc":
                        last_g = full_detach(model.fc.weight.grad)
                        last_g_bias = full_detach(model.fc.bias.grad)
                    elif nargs.model_name == "fcfc":
                        hidden_g = full_detach(model.fc.fc1.weight.grad)
                        hidden_g_bias = full_detach(model.fc.fc1.bias.grad)
                        last_g = full_detach(model.fc.fc2.weight.grad)
                        last_g_bias = full_detach(model.fc.fc2.bias.grad)
                    else:
                        print("Unimplemented model name")
                        break

                    # subtract one from other for last fc layer grads
                    last_g = last_g[0, :] - last_g[1, :]
                    last_g_bias = last_g_bias[0] - last_g_bias[1]
                    last_g = np.append(last_g, last_g_bias)
                    # print("last layer grad shape", last_g.shape)

                    if nargs.model_name == "fcfc":
                        # flatten hidden layer grads
                        hidden_g = np.append(hidden_g, np.reshape(hidden_g_bias, (hidden_g_bias.size, -1)), axis=1)
                        hidden_g = hidden_g.flatten()
                        # print("hidden layer grad shape", hidden_g.shape)

                        # append two layers gradients together
                        last_g = np.append(hidden_g, last_g)
                        # print("final grad shape", last_g.shape)

                    gs.append(last_g)
                    with torch.no_grad():
                        model.zero_grad()

    if get_grads:
        gs = np.stack(gs, axis=0)
        print("Gradients have shape", gs.shape)
        np.save(nargs.extracted_dir + "weight_bias" + nargs.model_name + "_grads_" + which + ".npy", gs)

    all_embed, all_y, all_g, all_i = np.array(all_embed), np.array(all_y), np.array(all_g), np.array(all_i)
    np.save(nargs.extracted_dir + which + "_data_resnet_" + nargs.model_name + ".npy", all_embed)
    np.save(nargs.extracted_dir + which + "_data_y_resnet_" + nargs.model_name + ".npy", all_y)
    np.save(nargs.extracted_dir + which + "_data_g_resnet_" + nargs.model_name + ".npy", all_g)
    np.save(nargs.extracted_dir + which + "_data_l_resnet_" + nargs.model_name + ".npy", all_l)
    np.save(nargs.extracted_dir + which + "_data_i_resnet_" + nargs.model_name + ".npy", all_i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pretrained-50")
    parser.add_argument("--alt_model_dir", type=str, default=None)
    parser.add_argument("--extracted_dir", type=str, default="extracted/")
    parser.add_argument("--get_fc_grads", default=False, action=argparse.BooleanOptionalAction)

    args_p = parser.parse_args()

    # os.putenv("CUDA_VISIBLE_DEVICES", "1")
    device = torch.device("cuda")

    if args_p.model_name == "pretrained-50":
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif args_p.model_name == "pretrained-18":
        model = torchvision.models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        # model_dir = "results/CUB/CUB_resnet_fc_grads/ERM_upweight_0_epochs_300_lr_1e-05_weight_decay_1.0/model_outputs/"
        # model = torch.load(model_dir + "160_model.pth")
        model = torch.load(args_p.alt_model_dir)

    model.eval()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    get_data(args_p, "train", get_grads=args_p.get_fc_grads)
    get_data(args_p, "val", get_grads=args_p.get_fc_grads)
    get_data(args_p, "test", get_grads=args_p.get_fc_grads)


