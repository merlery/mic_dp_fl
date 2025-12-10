import torch
from torchvision.models import alexnet, resnet18
from torch.nn.functional import relu, softmax, max_pool2d
from torch.nn.utils import spectral_norm
from torch import nn, tanh
import copy
from opacus.grad_sample import register_grad_sampler
from typing import Dict, Optional
import torchvision
from collections import OrderedDict
from numpy import median
import numpy as np
import torch.nn.functional as func
try:
    from mic_utils import compute_mic_weights
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    # This is OK - we'll use default initialization

def agg_weights(weights):
    with torch.no_grad():
        weights_avg = copy.deepcopy(weights[0])
        for k in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[k] += weights[i][k]
            weights_avg[k] = torch.div(weights_avg[k], len(weights))
    return weights_avg

def evaluate_global(users, test_dataloders, users_index):
    testing_corrects = 0
    testing_sum = 0
    for index in users_index:
        corrects, num = users[index].evaluate(test_dataloders[index])
        testing_corrects += corrects
        testing_sum += num
    print(f"Acc: {testing_corrects / testing_sum}")
    return (testing_corrects / testing_sum)


class InputNorm(nn.Module):
    def __init__(self, num_channel, num_feature):
        super().__init__()
        self.num_channel = num_channel
        self.gamma = nn.Parameter(torch.ones(num_channel))
        self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma*x
            x = x + self.beta
            return  x  
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta


class MICNorm(nn.Module):
    """
    MIC-based Input Normalization layer.
    Uses Maximum Information Coefficient to initialize transformation weights.
    """
    def __init__(self, num_channel, num_feature, mic_gamma: Optional[torch.Tensor] = None, 
                 mic_beta: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_channel = num_channel
        self.num_feature = num_feature
        
        # Initialize with MIC-based weights if provided, otherwise use default
        if mic_gamma is not None:
            self.gamma = nn.Parameter(mic_gamma.clone().detach())
        else:
            self.gamma = nn.Parameter(torch.ones(num_channel))
        
        if mic_beta is not None:
            self.beta = nn.Parameter(mic_beta.clone().detach())
        else:
            self.beta = nn.Parameter(torch.zeros(num_channel, num_feature, num_feature))
    
    def forward(self, x):
        if self.num_channel == 1:
            x = self.gamma * x
            x = x + self.beta
            return x
        if self.num_channel == 3:
            return torch.einsum('...ijk, i->...ijk', x, self.gamma) + self.beta
    
    def update_with_mic(self, data_batch: torch.Tensor, labels_batch: torch.Tensor):
        """
        Update transformation parameters based on MIC computation.
        This allows per-client personalization using MIC.
        """
        if not MIC_AVAILABLE:
            return
        
        try:
            # Flatten data for MIC computation
            if len(data_batch.shape) > 2:
                data_flat = data_batch.view(data_batch.size(0), -1).cpu().numpy()
            else:
                data_flat = data_batch.cpu().numpy()
            
            labels_np = labels_batch.cpu().numpy()
            
            # Compute MIC-based weights
            gamma_np, beta_np = compute_mic_weights(data_flat, labels_np)
            
            # Reshape and update parameters
            if self.num_channel == 1:
                # For 1 channel, gamma is a scalar or 1D
                if len(gamma_np.shape) == 0 or gamma_np.shape[0] == 1:
                    self.gamma.data = torch.from_numpy(gamma_np).float().to(self.gamma.device)
                else:
                    # Average if multiple values
                    self.gamma.data = torch.tensor(gamma_np.mean()).float().to(self.gamma.device)
            else:
                # For 3 channels, reshape appropriately
                if len(gamma_np) >= self.num_channel:
                    self.gamma.data = torch.from_numpy(gamma_np[:self.num_channel]).float().to(self.gamma.device)
        except Exception as e:
            print(f"Warning: Failed to update MIC weights: {e}")
        
class resnet18(torch.nn.Module):
    """Constructs a ResNet-18 model.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
    def forward(self, x):
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)

class resnet18_IN(torch.nn.Module):
    """Constructs a ResNet-18wIN model.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
        if num_classes == 8:
            self.norm = InputNorm(3, 150)
        else:
            self.norm = InputNorm(3, 120)
    def forward(self, x):
        x = self.norm(x)
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)


class resnet18_MIC(torch.nn.Module):
    """Constructs a ResNet-18 with MIC-based transformation.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
        if num_classes == 8:
            self.norm = MICNorm(3, 150)
        else:
            self.norm = MICNorm(3, 120)
    def forward(self, x):
        x = self.norm(x)
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)

class alexnet(torch.nn.Module):
    """Constructs a alexnet model.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.alexnet(pretrained=True)
        n_ftrs = self.backbone.classifier[-1].out_features
        self.fc = torch.nn.Linear(n_ftrs, num_classes)
    def forward(self, x):
        logits = self.backbone(x)
        logits = self.fc(logits)
        return logits, softmax(logits, dim=-1)


class alexnet_IN(torch.nn.Module):
    """Constructs a alexnet w IN model.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.alexnet(pretrained=True)
        n_ftrs = self.backbone.classifier[-1].out_features
        self.fc = torch.nn.Linear(n_ftrs, num_classes)
        self.norm = InputNorm(3, 150)
    def forward(self, x):
        x = self.norm(x)
        logits = self.backbone(x)
        logits = self.fc(logits)
        return logits, softmax(logits, dim=-1)


class alexnet_MIC(torch.nn.Module):
    """Constructs an alexnet with MIC-based transformation.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.alexnet(pretrained=True)
        n_ftrs = self.backbone.classifier[-1].out_features
        self.fc = torch.nn.Linear(n_ftrs, num_classes)
        self.norm = MICNorm(3, 150)
    def forward(self, x):
        x = self.norm(x)
        logits = self.backbone(x)
        logits = self.fc(logits)
        return logits, softmax(logits, dim=-1)


class mnist_fully_connected_IN(nn.Module):
    def __init__(self,num_classes):
        super(mnist_fully_connected_IN, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.norm = InputNorm(1, 28)

    def forward(self,x):
        x = self.norm(x)
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        logits = self.fc3(x)
        return logits, softmax(logits,dim=1)


class mnist_fully_connected_MIC(nn.Module):
    """MNIST model with MIC-based transformation (baseline comparison)"""
    def __init__(self, num_classes):
        super(mnist_fully_connected_MIC, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.norm = MICNorm(1, 28)

    def forward(self, x):
        x = self.norm(x)
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        logits = self.fc3(x)
        return logits, softmax(logits, dim=1)

class mnist_fully_connected(nn.Module):
    def __init__(self,num_classes):
        super(mnist_fully_connected, self).__init__()
        self.hidden1 = 600
        self.hidden2 = 100
        self.fc1 = nn.Linear(28 * 28, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc3 = nn.Linear(self.hidden2, num_classes, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        logits = self.fc3(x)
        return logits, softmax(logits,dim=1)

class purchase_fully_connected(nn.Module):
    def __init__(self,num_classes):
        super(purchase_fully_connected, self).__init__()
        self.fc1 = nn.Linear(600, 512,bias=False)
        self.fc2 = nn.Linear(512, 256,bias=False)
        self.fc3 = nn.Linear(256, 128,bias=False)
        self.fc4 = nn.Linear(128, num_classes,bias=False)


    def forward(self,x):
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = tanh(self.fc3(x))
        logits = self.fc4(x)
        return logits, softmax(logits,dim=1)

class purchase_fully_connected_IN(nn.Module):
    def __init__(self,num_classes):
        super(purchase_fully_connected_IN, self).__init__()
        self.fc1 = nn.Linear(600,512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.fc4 = nn.Linear(128, num_classes, bias=False)
        self.norm = FeatureNorm(600)
    def forward(self,x):
        x = self.norm(x)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = tanh(self.fc3(x))
        logits = self.fc4(x)
        return logits, softmax(logits,dim=1)


class purchase_fully_connected_MIC(nn.Module):
    """Purchase model with MIC-based transformation."""
    def __init__(self, num_classes):
        super(purchase_fully_connected_MIC, self).__init__()
        self.fc1 = nn.Linear(600, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.fc4 = nn.Linear(128, num_classes, bias=False)
        self.norm = FeatureNorm_MIC(600)
    def forward(self, x):
        x = self.norm(x)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = tanh(self.fc3(x))
        logits = self.fc4(x)
        return logits, softmax(logits, dim=1)

class linear_model(nn.Module):
    def __init__(self, num_classes, input_shape=512):
        super(linear_model, self).__init__()
        self.fc1 = nn.Linear(input_shape, num_classes, bias=True)
    def forward(self,x):
        logits = self.fc1(x)
        return logits, softmax(logits,dim=1)


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats
    bn_mean, bn_var = bn_mean.to(x.device), bn_var.to(x.device)
    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.reshape(view)) / torch.sqrt(bn_var.reshape(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.reshape(view) != 0)
    return x

class linear_model_DN(nn.Module):
    def __init__(self, num_classes, input_shape=512, bn_stats=False):
        super(linear_model_DN, self).__init__()
        if not bn_stats:
            self.bn_stats = (torch.zeros(input_shape), torch.ones(input_shape))
        else:
            mean = np.load('transfer/cifar100_resnext_mean.npy')
            var = np.load('transfer/cifar100_resnext_var.npy')
            self.bn_stats = (torch.from_numpy(mean), torch.from_numpy(var))
        self.fc1 = nn.Linear(input_shape, num_classes, bias=True)
    def forward(self,x):
        x = standardize(x, self.bn_stats)
        logits = self.fc1(x)
        return logits, softmax(logits,dim=1)


class FeatureNorm(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1, feature_shape))
    

    def forward(self, x):
        x = torch.einsum('ni, j->ni', x, self.gamma) 
        x = x + self.beta
        return  x


class FeatureNorm_MIC(nn.Module):
    """
    MIC-based Feature Normalization layer.
    Uses Maximum Information Coefficient to initialize transformation weights.
    """
    def __init__(self, feature_shape, mic_gamma: Optional[torch.Tensor] = None,
                 mic_beta: Optional[torch.Tensor] = None):
        super().__init__()
        self.feature_shape = feature_shape
        
        # Initialize with MIC-based weights if provided
        if mic_gamma is not None:
            self.gamma = nn.Parameter(mic_gamma.clone().detach())
        else:
            self.gamma = nn.Parameter(torch.ones(1))
        
        if mic_beta is not None:
            self.beta = nn.Parameter(mic_beta.clone().detach())
        else:
            self.beta = nn.Parameter(torch.zeros(1, feature_shape))
    
    def forward(self, x):
        x = torch.einsum('ni, j->ni', x, self.gamma)
        x = x + self.beta
        return x
    
    def update_with_mic(self, data_batch: torch.Tensor, labels_batch: torch.Tensor):
        """
        Update transformation parameters based on MIC computation.
        """
        if not MIC_AVAILABLE:
            return
        
        try:
            data_np = data_batch.cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            
            # Compute MIC-based weights
            gamma_np, beta_np = compute_mic_weights(data_np, labels_np)
            
            # Update parameters
            self.gamma.data = torch.tensor(gamma_np.mean()).float().to(self.gamma.device)
            if len(beta_np) == self.feature_shape:
                self.beta.data = torch.from_numpy(beta_np).float().reshape(1, -1).to(self.beta.device)
        except Exception as e:
            print(f"Warning: Failed to update MIC weights: {e}")


@register_grad_sampler(FeatureNorm)
def compute_grad_sample(
    layer: InputNorm, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    gs = torch.einsum("nk,nk->n", backprops, activations)
    ret = {layer.gamma: gs}
    if layer.beta is not None:
        ret[layer.beta] = torch.einsum("n...i->ni", backprops)

    return ret

class linear_model_DN_IN(nn.Module):
    def __init__(self, num_classes, input_shape, bn_stats=False):
        super(linear_model_DN_IN, self).__init__()
        if not bn_stats:
            self.bn_stats = (torch.zeros(input_shape), torch.ones(input_shape))
        else:
            mean = np.load('cifar100_resnext_mean.npy')
            var = np.load('cifar100_resnext_mean.npy')
            self.bn_stats = (torch.from_numpy(mean), torch.from_numpy(var))
        self.backbone = nn.Linear(input_shape, num_classes, bias=True)
        self.norm = FeatureNorm(input_shape)
    def forward(self,x):
        x = self.norm(x)
        x = standardize(x, self.bn_stats)
        logits = self.backbone(x)
        return logits, softmax(logits,dim=1)


class linear_model_DN_MIC(nn.Module):
    """Linear model with Data Normalization and MIC-based transformation."""
    def __init__(self, num_classes, input_shape, bn_stats=False):
        super(linear_model_DN_MIC, self).__init__()
        if not bn_stats:
            self.bn_stats = (torch.zeros(input_shape), torch.ones(input_shape))
        else:
            try:
                mean = np.load('transfer/cifar100_resnext_mean.npy')
                var = np.load('transfer/cifar100_resnext_var.npy')
                self.bn_stats = (torch.from_numpy(mean), torch.from_numpy(var))
            except:
                self.bn_stats = (torch.zeros(input_shape), torch.ones(input_shape))
        self.backbone = nn.Linear(input_shape, num_classes, bias=True)
        self.norm = FeatureNorm_MIC(input_shape)
    def forward(self, x):
        x = self.norm(x)
        x = standardize(x, self.bn_stats)
        logits = self.backbone(x)
        return logits, softmax(logits, dim=1)

@register_grad_sampler(InputNorm)
def compute_grad_sample(
    layer: InputNorm, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    gs = torch.einsum("nk...,nk...->nk", backprops, activations)
    ret = {layer.gamma: gs}
    if layer.beta is not None:
        ret[layer.beta] = torch.einsum("nijk->nijk", backprops)

    return ret


@register_grad_sampler(MICNorm)
def compute_grad_sample_mic(
    layer: MICNorm, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for MICNorm layer (same as InputNorm)
    """
    gs = torch.einsum("nk...,nk...->nk", backprops, activations)
    ret = {layer.gamma: gs}
    if layer.beta is not None:
        ret[layer.beta] = torch.einsum("nijk->nijk", backprops)
    return ret


@register_grad_sampler(FeatureNorm_MIC)
def compute_grad_sample_feature_mic(
    layer: FeatureNorm_MIC, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for FeatureNorm_MIC layer
    """
    gs = torch.einsum("nk,nk->n", backprops, activations)
    ret = {layer.gamma: gs}
    if layer.beta is not None:
        ret[layer.beta] = torch.einsum("n...i->ni", backprops)
    return ret
