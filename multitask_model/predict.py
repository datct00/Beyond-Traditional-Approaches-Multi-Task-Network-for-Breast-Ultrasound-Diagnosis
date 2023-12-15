import sys 
sys.path.append("..")
from dataloader import val_set
from model import multitask_model
from lib import torch, plt, F, T, cv2
from hyper import * 
from utils import UnNormalize, calculate_overlap_metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import random 
model = multitask_model().to(DEVICE)
model.load_state_dict(torch.load(WEIGHT,map_location=DEVICE)['model_state_dict'])



for _ in range(5):
    idx = random.randint(0,100)
    image, mask, label = val_set[idx]

    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    model.eval()
    with torch.no_grad(): 
        o_seg, o_class = model(image)



    #unnorm image
    unnorm_image = UnNormalize()(image).squeeze().permute(1,2,0).detach().cpu().numpy()

    #o_class (3) -> ()
    o_class = o_class.argmax(1).item()

    #o_seg (2,224,224) -> (224,224)
    o_seg = F.sigmoid(o_seg).round().argmax(dim=1)* 255
    o_seg = o_seg.permute(1,2,0).detach().cpu().numpy()

    

    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title(f"Input image")
    plt.imshow(unnorm_image)

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title(f"GT mask \n GT class: {CLASSES[label]}")
    plt.imshow(mask.argmax(0).detach().cpu().numpy())

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title(f"Pred mask \n Pred class: {CLASSES[o_class]}")
    plt.imshow(o_seg)

    plt.show()










