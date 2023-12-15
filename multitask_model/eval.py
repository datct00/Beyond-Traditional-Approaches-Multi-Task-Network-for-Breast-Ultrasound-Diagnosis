#Evaluation
from dataloader import train_loader, val_loader
from hyper import *
from model import multitask_model
from lib import logging, torch, os, smp, imp, DataLoader, Accuracy, optim, Precision, Recall, FBetaScore, T, F, Dice, tqdm, JaccardIndex
from utils import AverageMeter




WEIGHT = r"W:\breast_ultrasound\multitask_model\weight\resnet50_unet\best_model_1_f1=0.395.pth"


#Model
model = multitask_model().to(DEVICE)
model.load_state_dict(torch.load(WEIGHT, map_location=DEVICE))
model.eval()
print("Evaluating...")
progress = tqdm(val_loader, total=int(len(val_loader)))



#Loss & Optimizer
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
CE_loss = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)


#Metric Segmentation: IoU sorce, Dice score, Precision, Recall, F1 score
seg_dice_fn = Dice(num_classes=SEG_NUM_CLASSES, average='macro').to(DEVICE)
seg_iou_fn = JaccardIndex(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
seg_precision_fn = Precision(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
seg_recall_fn = Recall(num_classes=SEG_NUM_CLASSES, task='binary', average='macro').to(DEVICE)
seg_f1_score_fn = FBetaScore(num_classes=SEG_NUM_CLASSES, task="binary", beta=1.0).to(DEVICE)

#Metric Classification: Precision, Recall, F1 score
cla_acc_fn = Accuracy(num_classes=CLA_NUM_CLASSES, task="multiclass").to(DEVICE)
cla_precision_fn = Precision(num_classes=CLA_NUM_CLASSES, task='multiclass', average='macro').to(DEVICE)
cla_recall_fn = Recall(num_classes=CLA_NUM_CLASSES, task='multiclass', average='macro').to(DEVICE)
cla_f1_score_fn = FBetaScore(num_classes=CLA_NUM_CLASSES, task="multiclass", beta=1.0).to(DEVICE)


#Common meter
mean_f1_score_meter = AverageMeter()

#Meters segmentation
seg_iou_meter = AverageMeter()
seg_dice_meter = AverageMeter()
seg_precision_meter = AverageMeter()
seg_recall_meter = AverageMeter()
seg_f1_score_meter = AverageMeter()

#Meters classification
cla_acc_meter = AverageMeter()
cla_precision_meter = AverageMeter()
cla_recall_meter = AverageMeter()
cla_f1_score_meter = AverageMeter()

def eval():
    with torch.no_grad():
        #Reset meters
        #Common meter
        mean_f1_score_meter.reset()

        #Meters segmentation
        seg_iou_meter.reset()
        seg_dice_meter.reset()
        seg_precision_meter.reset()
        seg_recall_meter.reset()
        seg_f1_score_meter.reset()

        #Meters classification
        cla_acc_meter.reset()
        cla_precision_meter.reset()
        cla_recall_meter.reset()
        cla_f1_score_meter.reset()
        for batch_idx, (image, mask, label) in enumerate(progress):
            progress.refresh()
            n = image.shape[0]
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            label = label.to(DEVICE)

            #Forward
            output_mask, output_classification = model(image)


            #Calculate metrics
            #Segmentation: iou, dice, p, r, f1
            seg_iou_score = seg_iou_fn(output_mask, mask)
            seg_dice_output = F.sigmoid(output_mask.clone()).round()
            seg_dice_score = seg_dice_fn(seg_dice_output.long(), mask.long())
            seg_precision_score = seg_precision_fn(output_mask, mask)
            seg_recall_score = seg_recall_fn(output_mask, mask)
            seg_f1_score = seg_f1_score_fn(output_mask, mask)


            #Classification: acc, p, r, f1
            cla_acc = cla_acc_fn(output_classification,label)
            cla_precision_score = cla_precision_fn(output_classification, label)
            cla_recall_score = cla_recall_fn(output_classification, label)
            cla_f1_score = cla_f1_score_fn(output_classification, label)

            #Update meters

            #Segmentation
            seg_iou_meter.update(seg_iou_score.item(), n)
            seg_dice_meter.update(seg_dice_score.item(), n)
            seg_precision_meter.update(seg_precision_score.item(), n)
            seg_recall_meter.update(seg_recall_score.item(), n)
            seg_f1_score_meter.update(seg_f1_score.item(), n)

            #Classification
            cla_acc_meter.update(cla_acc.item(),n)
            cla_precision_meter.update(cla_precision_score.item(), n)
            cla_recall_meter.update(cla_recall_score.item(), n)
            cla_f1_score_meter.update(cla_f1_score.item(), n)

            #Common
            mean_f1_score = (cla_f1_score + seg_f1_score)/2
            mean_f1_score_meter.update(mean_f1_score.item(), n)
        print(f"Evaluation Result: Mean F1 Score: {mean_f1_score_meter.avg:.3f}")
        print(f"Classification: Accuracy: {cla_acc_meter.avg:.3f}, F1-Score: {cla_f1_score_meter.avg:.3f}, Precision: {cla_precision_meter.avg:.3f}, Recall: {cla_recall_meter.avg:.3f}")
        print(f"Segmentation: IoU: {seg_iou_meter.avg:.3f} Dice: {seg_dice_meter.avg:.3f}, F1-score: {seg_f1_score_meter.avg:.3f}, Precision: {seg_precision_meter.avg:.3f}, Recall: {seg_recall_meter.avg:.3f}")

if __name__ == "__main__":
    eval()