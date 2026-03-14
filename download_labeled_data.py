cddimport os
import zipfile
import shutil

os.environ['KAGGLE_USERNAME'] = 'aslnuraksakal'
os.environ['KAGGLE_KEY'] = '****'
import kaggle
from tqdm import tqdm


DATASET_NAME = "hoangxuanviet/multiclass-brain-hemorrhage-segmentation"
OUTPUT_DIR = "yeni_nifti_dataset/labeled_data" 

def main():
    print(f"Etiketli Veri İndirme Operasyonu Başlıyor...")
    print(f"Hedef: {DATASET_NAME}")
    
   
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        
        print("\n⬇Zip dosyası indiriliyor")
        kaggle.api.dataset_download_files(DATASET_NAME, path=".", unzip=False, quiet=False)
        
        zip_name = "multiclass-brain-hemorrhage-segmentation.zip"
        if not os.path.exists(zip_name):
            
            candidates = [f for f in os.listdir('.') if f.endswith('.zip')]
            if candidates:
                zip_name = candidates[0]
            else:
                print(" Zip dosyası bulunamadı!")
                return

        print(f" İndirme bitti: {zip_name}")

        
        print("\n📦 İçerik taranıyor ve etiketli veriler ayıklanıyor...")
        
        with zipfile.ZipFile(zip_name, 'r') as z:
            all_files = z.namelist()
            
            target_files = [
                f for f in all_files 
                if ("train" in f.lower() or "label" in f.lower()) 
                and "unlabel" not in f.lower()
            ]
            
            print(f"   -> Toplam {len(all_files)} dosyadan {len(target_files)} tanesi etiketli veri adayı.")
            print("   -> Çıkarma işlemi başlıyor...")
            
            for file in tqdm(target_files):
                z.extract(file, OUTPUT_DIR)
                
        print(f" Dosyalar '{OUTPUT_DIR}' klasörüne çıkarıldı.")
        
        
        print("\n🧹 Zip dosyası siliniyor (Disk tasarrufu)...")
        os.remove(zip_name)
        print(" Temizlik tamamlandı.")
        

        print("\n İndirilen Klasörler:")
        for root, dirs, files in os.walk(OUTPUT_DIR):
            level = root.replace(OUTPUT_DIR, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/ ({len(files)} dosya)")

    except Exception as e:
        print(f" HATA: {e}")

if __name__ == "__main__":
    main()