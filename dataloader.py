from dataset import BUSI 
from hyper import * 
from lib import DataLoader, T


transformer = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ])

train_set = BUSI(DATASET_DIR, input_size=INPUT_SIZE,transform=transformer, target_transform=None, is_train=True)
val_set = BUSI(DATASET_DIR, input_size=INPUT_SIZE,transform=transformer, target_transform=None, is_train=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
