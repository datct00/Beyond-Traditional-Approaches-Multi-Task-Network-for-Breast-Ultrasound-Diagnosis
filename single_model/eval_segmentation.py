#Evaluation
import sys 
sys.path.append("..")
from dataloader import val_loader
from hyper import *
from model import segmentation_model
from lib import torch, Precision, Recall, FBetaScore, tqdm, Dice, JaccardIndex, F
from utils import AverageMeter


from utils import calculate_overlap_metrics

#model 
model = segmentation_model().to(DEVICE)
model.load_state_dict(torch.load(WEIGHT, map_location=DEVICE)['model_state_dict'])
model.eval()
print("Evaluating...")
progress = tqdm(val_loader, total=int(len(val_loader)))


#Metric: IoU sorce, Dice score, Precision, Recall, F1 score
dice_fn = Dice(num_classes=SEG_NUM_CLASSES, average='macro').to(DEVICE)
iou_fn = JaccardIndex(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
precision_fn = Precision(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
recall_fn = Recall(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
f1_score_fn = FBetaScore(num_classes=SEG_NUM_CLASSES, task="binary", beta=1.0).to(DEVICE)

#Meters
iou_meter = AverageMeter()
dice_meter = AverageMeter()
precision_meter = AverageMeter()
recall_meter = AverageMeter()
f1_score_meter = AverageMeter()


sk_iou_meter = AverageMeter()
sk_dice_meter = AverageMeter()
sk_precision_meter = AverageMeter()
sk_recall_meter = AverageMeter()

def eval():
    with torch.no_grad():
        iou_meter.reset()
        dice_meter.reset()
        precision_meter.reset()
        recall_meter.reset()
        f1_score_meter.reset()


        sk_iou_meter.reset()
        sk_dice_meter.reset()
        sk_precision_meter.reset()
        sk_recall_meter.reset() 


        for batch_idx, (image, mask,_) in enumerate(progress):
            n = image.shape[0]
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            output = model(image)

            #Calculate metrics
            #IoU
            iou_score = iou_fn(output, mask)

            #Dice score
            dice_output = F.sigmoid(output.clone()).round()
            dice_score = dice_fn(dice_output.long(), mask.long())

            #P, R and F1
            precision_score = precision_fn(output, mask)
            recall_score = recall_fn(output, mask)
            f1_score = f1_score_fn(output, mask)

            sk_iou_score, sk_dice_score, sk_precision_score, sk_recall_score = calculate_overlap_metrics(output, mask)

            #Update meters
            iou_meter.update(iou_score.item(), n)
            dice_meter.update(dice_score.item(), n)
            precision_meter.update(precision_score.item(), n)
            recall_meter.update(recall_score.item(), n)
            f1_score_meter.update(f1_score.item(), n)

            sk_iou_meter.update(sk_iou_score, n)
            sk_dice_meter.update(sk_dice_score, n)
            sk_precision_meter.update(sk_precision_score, n)
            sk_recall_meter.update(sk_recall_score, n)


        print(f"\nEvaluation result: IoU: {iou_meter.avg:.3f}, Dice Score: {dice_meter.avg:.3f}, F1-Score: {f1_score_meter.avg:.3f}, Precision: {precision_meter.avg:.3f}, Recall: {recall_meter.avg:.3f}")
        print(f"\nQuans func result: IoU: {sk_iou_meter.avg:.3f}, Dice Score: {sk_dice_meter.avg:.3f}, F1-Score: {f1_score_meter.avg:.3f}, Precision: {sk_precision_meter.avg:.3f}, Recall: {sk_recall_meter.avg:.3f}")




if __name__ == "__main__":
    eval()