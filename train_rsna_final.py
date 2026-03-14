import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import time
import copy
import numpy as np

IMAGE_FOLDER = 'png_dataset'
CSV_FILE = 'ham_veri/final_labels.csv'
PRETRAINED_PATH = 'best_rsna_resnet50_v2.pth'
SAVE_PATH = 'best_rsna_6class_final.pth'
BATCH_SIZE = 16 
NUM_EPOCHS = 25
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
CLASS_NAMES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

class RSNAMultiLabelDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.labels_dict = {}
        for _, row in self.df.iterrows():
            lbls = row[CLASS_NAMES].values.astype('float32')
            self.labels_dict[row['StudyInstanceUID']] = lbls
            
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.valid_files = []
        for f in self.image_files:
            sid = f.rsplit('_', 1)[0]
            if sid in self.labels_dict:
                self.valid_files.append(f)
                
        print(f"Dataset Hazır: {len(self.valid_files)} resim işlenecek.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        study_id = filename.rsplit('_', 1)[0]
        label = self.labels_dict[study_id]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def calculate_accuracy(output, target):
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).float()
    correct = (preds == target).float().sum()
    total = torch.numel(target)
    return correct / total

def main():
    print(f"6-Sınıflı Final Eğitim Başlıyor...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = RSNAMultiLabelDataset(IMAGE_FOLDER, CSV_FILE, transform=data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_PATH))
        print("Önceki Binary Model Başarıyla Yüklendi!")
    except:
        print("Binary model bulunamadı, ImageNet ağırlıklarıyla sıfırdan başlanıyor.")
        model = models.resnet50(pretrained=True)
        
    model.fc = nn.Linear(num_ftrs, 6)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_val_loss = float('inf')
    
    print(f"\n{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^12} | {'Val Loss':^12} | {'Val Acc':^12} | {'Durum'}")
    print("-" * 75)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_batches_train = len(dataloaders['train'])
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += calculate_accuracy(outputs, labels).item() * inputs.size(0)
            
        scheduler.step()
        
        epoch_train_loss = running_loss / train_size
        epoch_train_acc = running_acc / train_size

        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_acc += calculate_accuracy(outputs, labels).item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / val_size
        epoch_val_acc = running_val_acc / val_size

        status = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            status = "KAYDEDİLDİ (En İyi Model)"
        else:
            status = "..."

        print(f"{epoch+1:^7} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^12.4f} | {epoch_val_loss:^12.4f} | {epoch_val_acc:^12.4f} | {status}")

    print("-" * 75)
    print(f"Eğitim Tamamlandı. En iyi model şurada: {SAVE_PATH}")
    print(f"En Düşük Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()