from model import classification_model, segmentation_model
from hyper import *
from lib import cv2, T, F, os, torch, plt, time
from utils import UnNormalize

def two_single_models(image, mask=None, label=None, plot=True, classification_weight=None, segmentation_weight=None):
    if isinstance(image, str):
        assert os.path.exists(image), "Image not found"
        image = cv2.imread(image)
        #Process image
        image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        image = T.ToTensor()(image)
        image = image.to(DEVICE)
        image = T.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.unsqueeze(0)
    elif isinstance(image,torch.Tensor):
        image = image.unsqueeze(0)
        image = image.to(DEVICE)
   
    # Load models
    class_model = classification_model().to(DEVICE)
    seg_model = segmentation_model().to(DEVICE)
    

    #Load weight
    if classification_weight is not None:
        assert os.path.exists(classification_weight), "Classification weight not found"
        class_model.load_state_dict(torch.load(classification_weight,map_location=DEVICE)['model_state_dict'])
    if segmentation_weight is not None:
        assert os.path.exists(segmentation_weight), "Segmentation weight not found"
        seg_model.load_state_dict(torch.load(segmentation_weight,map_location=DEVICE)['model_state_dict'])

    #Inference
    with torch.no_grad():
        class_model.eval() 
        seg_model.eval()
        s = time.time()
        print("Segmentation model inference...",end=" ")
        mask_output = seg_model(image)
        print(f"Done. Time: {round(time.time(),3)-s}s")
        print("Classification model inference...",end=" ")
        c = time.time()
        class_output = class_model(image)
        print(f"Done. Time: {round(time.time(),3)-c}s")


    #Post-processing output
    mask_output = F.sigmoid(mask_output).round().argmax(dim=1) * 255
    mask_output = mask_output.permute(1,2,0).detach().cpu().numpy()
    class_output = torch.argmax(class_output, dim=1).detach().cpu()
    class_output = CLASSES[class_output.item()]
    image = UnNormalize()(image)
    image = image.squeeze(0).permute(1,2,0).detach().cpu().numpy()

    mask = torch.argmax(mask,0) * 255 
    mask = mask.detach().cpu().numpy()
    
    #Plot
    if plot:
        if mask is not None and label is not None:
            plt.subplot(1,3,1)
            plt.imshow(image)
            plt.title("Input image")
            plt.axis("off")
            plt.subplot(1,3,2)
            plt.imshow(mask)
            plt.title("GT mask \n GT class: {}".format(CLASSES[label]))
            plt.axis("off")
            plt.subplot(1,3,3)
            plt.imshow(mask_output)
            plt.title("Pred mask \n Pred class: {}".format(class_output))
            plt.axis("off")
            plt.show()
        else:
            plt.subplot(1,2,1)
            plt.imshow(image)
            plt.title("Input image")
            plt.axis("off")
            plt.subplot(1,2,2)
            plt.imshow(mask_output)
            plt.title("Pred Mask - {}".format(class_output))
            plt.axis("off")
            plt.show()
    else:
        return mask_output, class_output

if __name__ == "__main__":
    from dataset import BUSI
    import random 
    
    transformer = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  ])
    dataset = BUSI(DATASET_DIR, input_size=INPUT_SIZE,transform=transformer, target_transform=None, is_train=False)


    for _ in range(5):
        idx = random.randint(0, 100)
        image, mask, label = dataset[idx]
        class_weight = r"W:\breast_ultrasound\single_model\weight\resnet50_classification\classification\best_50_448_BS=8_f1=0.822.pth"
        seg_weight = None #r"W:\breast_ultrasound\single_model\weight\resnet50_fpn\segmentation\best_1_448_BS=8_iou=0.872.pth"
        two_single_models(image, mask=mask, label=label, plot=True, classification_weight=class_weight, segmentation_weight=seg_weight)

    