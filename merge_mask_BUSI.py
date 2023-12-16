from lib import cv2, os
from hyper import DATASET_DIR
from collections import defaultdict

def merge_mask_BUSI_dataset(): 
    """
    Combine multiple masks into a single mask 
    """
    def combine_mask(*masks):
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask
    
    if not os.path.exists(DATASET_DIR):
        raise Exception("BUSI directory does not exist")
    
    benign_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "benign")) if f.endswith(".png")]
    malignant_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "malignant")) if f.endswith(".png")]
    normal_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "normal")) if f.endswith(".png")]

    benign = defaultdict(list)
    malignant = defaultdict(list)
    normal = defaultdict(list)
    
    for file in benign_files:
        if "_" in file:
            benign[file.split("_")[0]].append(file)
    for file in malignant_files:
        if "_" in file:
            malignant[file.split("_")[0]].append(file)
    for file in normal_files:
        if "_" in file:
            normal[file.split("_")[0]].append(file)
    
    assert len(benign) == 437, f"{len(benign)} != 473"
    assert len(malignant) == 210, f"{len(malignant)} != 210"
    assert len(normal) == 133, f"{len(normal)} != 133"
    
    #Combine masks 
    for image in benign:
        mask_length = len(benign[image])
        
        if mask_length == 1:
            continue
        elif mask_length > 1:
            masks_path = [os.path.join(DATASET_DIR, "benign", mask) for mask in benign[image]]

            masks_image = [cv2.cvtColor(
                        cv2.imread(
                            mask
                        ), 
                    cv2.COLOR_BGR2GRAY) for mask in masks_path]
            
            combined_mask = combine_mask(*masks_image)
            log_file_path = os.path.join(DATASET_DIR, "benign_combined_files.txt")
            with open(log_file_path, "a") as f:
                f.write(image + "|" + str(len(benign[image])) + "\n")
                
            #Remove the old masks
            for mask in masks_path:
                os.remove(mask)
                
            #Save the combined mask
            cv2.imwrite(os.path.join(DATASET_DIR, "benign", image + "_mask.png"), combined_mask)    
        else:
            raise Exception("Less then 1 mask")
    
    for image in malignant:
        mask_length = len(malignant[image])
        
        if mask_length == 1:
            continue
        elif mask_length > 1:
            masks_path = [os.path.join(DATASET_DIR, "malignant", mask) for mask in malignant[image]]

            masks_image = [cv2.cvtColor(
                        cv2.imread(
                            mask
                        ), 
                    cv2.COLOR_BGR2GRAY) for mask in masks_path]
            
            combined_mask = combine_mask(*masks_image)
            log_file_path = os.path.join(DATASET_DIR, "malignant_combined_files.txt")
            with open(log_file_path, "a") as f:
                f.write(image + "|" + str(len(malignant[image])) + "\n")
                
            #Remove the old masks
            for mask in masks_path:
                os.remove(mask)
                
            #Save the combined mask
            cv2.imwrite(os.path.join(DATASET_DIR, "malignant", image + "_mask.png"), combined_mask)    
        else:
            raise Exception("Less then 1 mask")
        
    for image in normal:
        mask_length = len(normal[image])
        
        if mask_length == 1:
            continue
        elif mask_length > 1:
            masks_path = [os.path.join(DATASET_DIR, "normal", mask) for mask in normal[image]]

            masks_image = [cv2.cvtColor(
                        cv2.imread(
                            mask
                        ), 
                    cv2.COLOR_BGR2GRAY) for mask in masks_path]
            
            combined_mask = combine_mask(*masks_image)
            log_file_path = os.path.join(DATASET_DIR, "normal_combined_files.txt")
            with open(log_file_path, "a") as f:
                f.write(image + "|" + str(len(normal[image])) + "\n")
                
            #Remove the old masks
            for mask in masks_path:
                os.remove(mask)
                
            #Save the combined mask
            cv2.imwrite(os.path.join(DATASET_DIR, "normal", image + "_mask.png"), combined_mask)    
        else:
            raise Exception("Less then 1 mask")
if __name__ == "__main__":
    merge_mask_BUSI_dataset()