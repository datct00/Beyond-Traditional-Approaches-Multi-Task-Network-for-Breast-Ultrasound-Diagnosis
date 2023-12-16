from hyper import DATASET_DIR
from dataloader import train_set, val_set


class TRAIN_SET: 
    TOTAL = 623
    BENIGN = 349
    MALIGNANT = 168
    NORMAL = 106

class VAL_SET:
    TOTAL = 157
    BENIGN = 88
    MALIGNANT = 42
    NORMAL = 27

#Check if the dataset is correct
    
def print_notice(message, status="[OK]", indent=4, width=40):
    # Calculate the remaining dots needed for consistent formatting
    remaining_dots = width - len(message)

    # Construct the notice string with consistent indentation
    notice = f"{message}{'.' * remaining_dots}{status}"

    # Print the notice
    print(notice)

def check(): 
    print("Checking dataset...")
    if len(train_set) != TRAIN_SET.TOTAL:
        print_notice(f"Trainset", status=f"[X] - Wrong number of images! Expected {TRAIN_SET.TOTAL} but got {len(train_set)}")
    else: 
        print_notice(f"Trainset", status=f"[OK]")
    if len(train_set.b_train_set) != TRAIN_SET.BENIGN:
        print_notice(f"Trainset - Bengin", status=f"[X] - Wrong number of images! Expected {TRAIN_SET.BENIGN} but got {len(train_set.b_train_set)}")
    else:
        print_notice(f"Trainset - Bengin", status=f"[OK]")
    if len(train_set.m_train_set) != TRAIN_SET.MALIGNANT:
        print_notice(f"Trainset - Malignant", status=f"[X] - Wrong number of images! Expected {TRAIN_SET.MALIGNANT} but got {len(train_set.m_train_set)}")
    else:
        print_notice(f"Trainset - Malignant", status=f"[OK]")
    if len(train_set.n_train_set) != TRAIN_SET.NORMAL:
        print_notice(f"Trainset - Normal", status=f"[X] - Wrong number of images! Expected {TRAIN_SET.NORMAL} but got {len(train_set.n_train_set)}")
    else:
        print_notice(f"Trainset - Normal", status=f"[OK]")
    if len(val_set) != VAL_SET.TOTAL:
        print_notice(f"Valset", status=f"[X] - Wrong number of images! Expected {VAL_SET.TOTAL} but got {len(val_set)}")
    else:
        print_notice(f"Valset", status=f"[OK]")
    if len(val_set.b_val_set) != VAL_SET.BENIGN:
        print_notice(f"Valset - Bengin", status=f"[X] - Wrong number of images! Expected {VAL_SET.BENIGN} but got {len(val_set.b_val_set)}")
    else:
        print_notice(f"Valset - Bengin", status=f"[OK]")
    if len(val_set.m_val_set) != VAL_SET.MALIGNANT:
        print_notice(f"Valset - Malignant", status=f"[X] - Wrong number of images! Expected {VAL_SET.MALIGNANT} but got {len(val_set.m_val_set)}")
    else:
        print_notice(f"Valset - Malignant", status=f"[OK]")
    if len(val_set.n_val_set) != VAL_SET.NORMAL:
        print_notice(f"Valset - Normal", status=f"[X] - Wrong number of images! Expected {VAL_SET.NORMAL} but got {len(val_set.n_val_set)}")
    else:
        print_notice(f"Valset - Normal", status=f"[OK]")
    print("Done!")

    

if __name__ == "__main__":
    check()

