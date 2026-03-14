import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

MODEL_PATH = 'rsna_adapted_new_hospital.pth'
IMG_DIR = 'unlabeled_png_data'
REPORT_CSV = 'hospital_triage_report.csv'

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
        return cam

def overlay_heatmap(img_path, cam):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.6
    return np.uint8(result)

def main():
    print("DOĞRULAMA: Modelin tespitleri görselleştiriliyor...")
    
    df = pd.read_csv(REPORT_CSV)
    
    sickest = df.sort_values(by='any', ascending=False).head(4)
    
    healthiest = df.sort_values(by='any', ascending=True).head(4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    grad_cam = GradCAM(model, model.layer4[2])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    plt.figure(figsize=(16, 9))
    
    for i, (_, row) in enumerate(sickest.iterrows()):
        fname = row['filename']
        prob = row['any']
        source = row['source_folder'] if 'source_folder' in row else 'Bilinmiyor'
        
        img_path = os.path.join(IMG_DIR, fname)
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        cam = grad_cam(input_tensor)
        vis = overlay_heatmap(img_path, cam)
        
        plt.subplot(2, 4, i+1)
        if vis is not None:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        
        title_color = 'green' if source == 'anybleed' else 'red'
        
        plt.title(f"Prediction: %{prob*100:.1f}\nTrue Value: {source}", color=title_color, fontweight='bold', fontsize=10)
        plt.axis('off')

    for i, (_, row) in enumerate(healthiest.iterrows()):
        fname = row['filename']
        prob = row['any']
        source = row['source_folder'] if 'source_folder' in row else 'Bilinmiyor'
        
        img_path = os.path.join(IMG_DIR, fname)
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        cam = grad_cam(input_tensor)
        vis = overlay_heatmap(img_path, cam)
        
        plt.subplot(2, 4, i+5)
        if vis is not None:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            
        title_color = 'green' if source == 'nobleed' else 'red'
        
        plt.title(f"\nPrediction: %{prob*100:.1f}\nTrue Value: {source}", color=title_color, fontweight='bold', fontsize=10)
        plt.axis('off')
        
    plt.suptitle("Bleeding/No Bleeding Visualization with GradCAM", fontsize=16)
    plt.tight_layout()
    plt.savefig('final_kanit.png')
    print("Görsel Kanıt Hazır: 'final_kanit.png'")

if __name__ == "__main__":
    main()