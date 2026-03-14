import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


MODEL_PATH = 'rsna_adapted_new_hospital.pth'
IMG_DIR = 'unlabeled_png_data'
FILE_LIST = 'unlabeled_file_list.csv'
OUTPUT_CSV = 'hospital_triage_report.csv'
BATCH_SIZE = 128
CLASS_NAMES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

class UnlabeledDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = self.df['filename'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image

def main():
    print("SANAL HASTANE TARAMASI BAŞLIYOR (vFinal)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Çalışma Ortamı: {device}")
    
    df = pd.read_csv(FILE_LIST)
    print(f"Toplam Dosya: {len(df)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = UnlabeledDataset(df, IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Model Yükleniyor...")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Model yüklenemedi! Yol doğru mu? ({MODEL_PATH})")
        return

    model = model.to(device)
    model.eval()
    all_probs = []
    
    print("Tarama Başladı (Bu işlem dosya sayısına göre 5-10 dk sürebilir)...")
    with torch.no_grad():
        for inputs in tqdm(loader, desc="Taranıyor"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            
    all_probs = np.vstack(all_probs)
    
    print("Sonuçlar Kaydediliyor...")
    for i, cls in enumerate(CLASS_NAMES):
        df[cls] = all_probs[:, i]
        
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nRAPOR HAZIRLANDI: {OUTPUT_CSV}")
    print("-" * 50)
    
    if 'source_folder' in df.columns:
        print("🕵️DOĞRULUK ÖN İZLEMESİ:")
        
        sick_group = df[df['source_folder'] == 'anybleed']
        healthy_group = df[df['source_folder'] == 'nobleed']
        
        avg_sick_score = sick_group['any'].mean()
        avg_healthy_score = healthy_group['any'].mean()
        
        print(f"   'anybleed' Klasörünün Ortalama Risk Puanı: %{avg_sick_score*100:.2f}")
        print(f"   'nobleed' Klasörünün Ortalama Risk Puanı : %{avg_healthy_score*100:.2f}")
        
        if avg_sick_score > avg_healthy_score:
            print("   MÜKEMMEL! Model hasta klasörüne çok daha yüksek puan vermiş.")
        else:
            print("   Dikkat: Puanlar birbirine yakın veya ters.")

if __name__ == "__main__":
    main()