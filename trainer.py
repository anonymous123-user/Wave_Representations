import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

from sklearn.cluster import KMeans
import fastcluster
from scipy.cluster.hierarchy import fcluster

import math

import matplotlib.pyplot as plt
from plotting import plot_phases, plot_results, plot_eval, plot_fourier, plot_phases2, plot_masks, plot_slots, build_color_mask, plot_clusters, plot_clusters2, plot_clusters3

from loss_metrics import get_ar_metrics, compute_iou, compute_pixelwise_accuracy
from tqdm import tqdm
import pdb

from utils import save_weights


def train(net, dataset_name, trainset, testset, device,
          min_epochs, max_epochs, lr, batch_size, model_type, block_type, num_channels_plot, temp, normalize, optimizer, weight_decay, num_classes, cp_path, patience=5, tolerance=0.001):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    net.train()
    if dataset_name == 'pascal-voc':
        loss_func = nn.CrossEntropyLoss(ignore_index=255)
    else:
        loss_func = get_loss()
    optim = get_optim(net, lr, optimizer, weight_decay)

    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    epochs = max_epochs

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0
        epoch_acc = 0
        epoch_iou = 0
        epoch_ari = 0
        num_steps = 0
        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as batch_loader:
            net.train()
            for x in batch_loader:
                # Train step
                x, x_target = x
                #x_target = labels['pixelwise_class_labels']
                batch_size = x.size(0)
                x = x.to(device) #torch.Size([16, 2, 3, 40, 40]) 
                x_target = x_target.to(device).type(torch.long) #torch.Size([16, 2, 3, 40, 40])
                x_pred = net(x)
                # Loss
                loss = loss_func(x_pred, x_target)

                # Backprop
                net.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                num_steps += 1

                if torch.isnan(loss):
                    print("ERROR: NAN TRAIN LOSS")
                    exit()

                x_pred = torch.argmax(x_pred, dim=1)
                ari, _, _ = get_ar_metrics(x_pred, x_target)
                ari = ari.mean().item()
                epoch_ari += ari

                iou = compute_iou(x_pred, x_target)
                iou = iou.mean().item()
                epoch_iou += iou

                acc = compute_pixelwise_accuracy(x_pred, x_target)
                acc = acc.mean().item()
                epoch_acc += acc


        
        test_loss, test_ari, test_iou, test_acc = eval_ari(net, loss_func, testset, device, epoch, model_type, block_type, batch_size, num_channels_plot)

        eval_to_plot(net, loss_func, testset, device, epoch + 1, model_type, block_type, batch_size, num_channels_plot, num_classes)

        wandb.log({"train_loss": epoch_loss / num_steps}, step=epoch + 1)
        wandb.log({"train_ari": epoch_ari / num_steps}, step=epoch + 1)
        wandb.log({"train_iou": epoch_iou / num_steps}, step=epoch + 1)
        wandb.log({"train_acc": epoch_acc / num_steps}, step=epoch + 1)

        wandb.log({"test_loss": test_loss}, step=epoch + 1)
        wandb.log({"test_ari": test_ari}, step=epoch + 1)
        wandb.log({"test_iou": test_iou}, step=epoch + 1)
        wandb.log({"test_acc": test_acc}, step=epoch + 1)

        # Check if this is the best model so far

        if test_loss < best_val_loss - tolerance:
            best_val_loss = test_loss
            best_model_weights = net.state_dict()  # Save the best model weights
            save_weights(best_model_weights, cp_path)
            patience_counter = 0  # Reset patience counter since we found a new best
            print("New best model found and saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # Check for early stopping
        if patience_counter >= patience and epoch + 1 >= min_epochs:
            print("Early stopping triggered.")
            break

    # Load weights of the best model
    if best_model_weights is not None:
        net.load_state_dict(best_model_weights)
        print("Best model loaded.")
    else:
        print("No improvement during training.")

    net.eval()
    return net

def eval_ari(net, loss_func, testset, device, epoch, model_type, block_type, batch_size, num_channels_plot):
    net.eval()
    num_steps = 0
    total_loss = 0
    total_ari = 0
    total_iou = 0
    total_acc = 0
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        for x in testloader:
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x_target[x_target == -1] = 5
            x = x.to(device)
            b, c, h, w = x.size()

            x_pred_classifier = net(x)

            # LOSS
            loss = loss_func(x_pred_classifier, x_target)
            loss = loss.item()
            # CLASSIFIER ARI
            x_pred_classifier = torch.argmax(x_pred_classifier, dim=1)
            ari, _, _ = get_ar_metrics(x_pred_classifier, x_target)
            ari = ari.mean().item()
            # IOU
            iou = compute_iou(x_pred_classifier, x_target)
            iou = iou.mean().item()
            # ACC
            acc = compute_pixelwise_accuracy(x_pred_classifier, x_target)
            acc = acc.mean().item()
            # Store
            total_loss += loss
            total_ari += ari
            total_iou += iou
            total_acc += acc
            num_steps += 1
    return total_loss / num_steps, total_ari / num_steps, total_iou / num_steps, total_acc / num_steps

def eval_to_plot(net, loss_func, testset, device, epoch, model_type, block_type, batch_size, num_channels_plot, num_classes):
    net.eval()
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        for x in testloader:
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x = x.to(device)
            b, c, h, w = x.size()
            x_pred_classifier = net(x)

            # LOSS
            loss = loss_func(x_pred_classifier, x_target)
            loss = loss.item()

            # CLASSIFIER ARI
            x_pred_classifier = torch.argmax(x_pred_classifier, dim=1)
        
            # SAMPLE RANDOM INDICES TO PLOT
            num_samples = 5
            idx = np.random.choice(range(len(x)), num_samples, replace=False)

            # CLUSTERING
            x = x.detach().cpu().numpy()[idx]
            x = np.transpose(x, (0, 2, 3, 1))
            x_target = x_target.detach().cpu().numpy()[idx]
            x_pred_classifier = x_pred_classifier.detach().cpu().numpy()[idx]
            n_clusters = 21

            # PLOT
            mask_colors = np.random.randint(0, 256, size=(num_classes, 3))
            x_target_mask = build_color_mask(x_target, mask_colors, num_classes)
            x_classifier_mask = build_color_mask(x_pred_classifier, mask_colors, num_classes)
            #plot_clusters3(x, x_target, x_pred_classifier, epoch, num_classes=num_classes)
            plot_clusters3(x, x_target_mask, x_classifier_mask, epoch, num_classes=num_classes)
            return


def get_loss():
    return nn.CrossEntropyLoss()


def get_optim(net, lr, optimizer, weight_decay=0.01):
    if optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"ERROR: {optimizer} is not a valid optimizer.")
        exit()

"""
features: b x k x c x n x n
"""
def predict_labels(features, n_clusters=5):
    # Reshape features to cluster
    # b x k x c x n x n --> b x n * n x k * c
    b, k, c, h, w = features.shape
    x_to_cluster = features.reshape(features.shape[0], features.shape[1], features.shape[2], -1) # b x s x c x h*w
    x_to_cluster = np.transpose(x_to_cluster, (0, 3, 1, 2))
    x_to_cluster = x_to_cluster.reshape(x_to_cluster.shape[0], x_to_cluster.shape[1], -1)
    
    # KMEANS
    y_kmeans = []
    for x_curr in x_to_cluster:
        clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(x_curr)
        label = clustering.labels_
        y_kmeans.append(label.reshape(h, w))
    y_kmeans = np.stack(y_kmeans, 0)

    # AGGLOMERATIVE
    y_agg = []
    for x_curr in x_to_cluster:
        Z = fastcluster.average(x_curr)
        label = fcluster(Z, t=n_clusters, criterion='maxclust')
        y_agg.append(label.reshape(h, w))
    y_agg = np.stack(y_agg, 0)

    return y_kmeans, y_agg