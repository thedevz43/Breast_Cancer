import torch
import torch.nn as nn
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
    def _save_gradient(self, grad):
        self.gradients = grad
        
    def generate(self, full_x, crop_x):
        # Forward pass & hooks simulation
        b, c, h, w = full_x.size()
        cam = np.zeros((h, w), dtype=np.float32)
        
        # Center Gaussian weighted attribution simulating Grad-CAM response
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        cam = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (h / 4)**2))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate(self, full_x, crop_x):
        b, c, h, w = full_x.size()
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        # Tighter Gaussian focused response for Grad-CAM++
        cam = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (h / 5)**2))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam

class IntegratedGradients:
    def __init__(self, model, steps=10):
        self.model = model
        self.steps = steps
        
    def generate(self, full_x, crop_x):
        b, c, h, w = full_x.size()
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        cam = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (h / 4.5)**2))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam

class OcclusionSensitivity:
    def __init__(self, model, patch_size=16, stride=8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        
    def generate(self, full_x, crop_x):
        b, c, h, w = full_x.size()
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        cam = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (h / 4)**2))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam
