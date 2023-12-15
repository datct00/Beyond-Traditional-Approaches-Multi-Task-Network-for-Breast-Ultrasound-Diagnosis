from lib import torch, os


NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Solver
CLASSES = {0: "Benign", 1: "Malignant", 2: "Normal"}
INPUT_SIZE = (448,448)
BATCH_SIZE = 8
BASE_LR = 0.001
MAX_EPOCHS = 50
SAVE_INTERVAL = 10
PATIENCE = 300


#Model
ARCH = "deeplabv3plus" # ['unet', 'unetpp', , 'fpn', 'deeplabv3plus']
ENCODER_NAME = "efficientnet-b4" # ['resnet50', 'resnext50_32x4d', 'tu-wide_resnet50_2', 'efficientnet-b4']
IN_CHANNELS = 3
SEG_NUM_CLASSES = 2
CLA_NUM_CLASSES = 3
OUTPUT_ACTIVATION = None #None for logits 

#Loss coefficient weight
ALPHA = 0.7

#Path
OUTPUT_DIR = r"W:\breast_ultrasound\multitask_model\weight"
DATASET_DIR = r"W:\breast_ultrasound\datasets\Dataset_BUSI_with_GT"
CHECKPOINT = None

#Eval
WEIGHT = r"W:\breast_ultrasound\multitask_model\weight\resnet50_unetpp\best_24_448_BS=8_f1=0.790.pth"