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
    print(f"Downloading the Labeled Data Set")
    print(f"Target: {DATASET_NAME}")
    
   
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        kaggle.api.dataset_download_files(DATASET_NAME, path=".", unzip=False, quiet=False)
        
        zip_name = "multiclass-brain-hemorrhage-segmentation.zip"
        if not os.path.exists(zip_name):
            
            candidates = [f for f in os.listdir('.') if f.endswith('.zip')]
            if candidates:
                zip_name = candidates[0]
            else:
                print("couldn't find the zip file")
                return

        print(f" downloaded: {zip_name}")
        
        with zipfile.ZipFile(zip_name, 'r') as z:
            all_files = z.namelist()
            
            target_files = [
                f for f in all_files 
                if ("train" in f.lower() or "label" in f.lower()) 
                and "unlabel" not in f.lower()
            ]
            

            for file in tqdm(target_files):
                z.extract(file, OUTPUT_DIR)
                
        print(f" Documents are at: '{OUTPUT_DIR}' ")
        
        
        print("\n Removing zip file")
        os.remove(zip_name)

        print("\n downloaded folders")
        for root, dirs, files in os.walk(OUTPUT_DIR):
            level = root.replace(OUTPUT_DIR, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/ ({len(files)} )")

    except Exception as e:
        print(f" error: {e}")

if __name__ == "__main__":
    main()
