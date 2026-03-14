import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

MODEL_PATH = 'rsna_adapted_new_hospital.pth'
TEST_FOLDER = 'test_png_data' 

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output)
        self.model.zero_grad()
        target = output[0][class_idx]
        target.backward()
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, class_idx

def overlay_heatmap(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.6
    return np.uint8(result)

def main():
    print("Yeni Modelin Dikkati (Grad-CAM) Görselleştiriliyor...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    grad_cam = GradCAM(model, model.layer4[2])
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    plt.figure(figsize=(10, 10))
    
    files = [f for f in os.listdir(TEST_FOLDER) if f.endswith('.png')]
    selected_files = random.sample(files, 4)
    
    for i, fname in enumerate(selected_files):
        img_path = os.path.join(TEST_FOLDER, fname)
        orig_pil = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(orig_pil).unsqueeze(0).to(device)
        
        cam, class_idx = grad_cam(input_tensor)
        result_img = overlay_heatmap(img_path, cam)
        
        plt.subplot(2, 2, i+1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Örnek {i+1}\nTahmin Sınıfı: {class_idx}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('adapted_heatmap.png')
    print("Yeni görselleştirme 'adapted_heatmap.png' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()