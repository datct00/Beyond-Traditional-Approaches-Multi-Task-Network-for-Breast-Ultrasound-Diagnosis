import sys 
sys.path.append("..")
from dataloader import train_loader, val_loader
from hyper import *
from model import multitask_model
from lib import time, torch, os, smp, accuracy_score, precision_score, recall_score, f1_score, optim, F, focal_loss
from utils import AverageMeter, setup_logger, logging_hyperparameters, init_path

#Task 
TASK = "multitask"

#Path 
weight_dir, log_dir, logger_name = init_path(TASK)


#Model
model = multitask_model().to(DEVICE)

#Loss & Optimizer
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
# CE_loss = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)



#Common meter 
overall_meter = AverageMeter()
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()

#Meters segmentation
seg_train_loss_meter = AverageMeter()
seg_val_loss_meter = AverageMeter()
seg_iou_meter = AverageMeter()
seg_dice_meter = AverageMeter()
seg_precision_meter = AverageMeter()
seg_recall_meter = AverageMeter()
seg_f1_score_meter = AverageMeter()

#Meters classification
cla_train_loss_meter = AverageMeter()
cla_val_loss_meter = AverageMeter()
cla_acc_meter = AverageMeter()
cla_precision_meter = AverageMeter()
cla_recall_meter = AverageMeter()
cla_f1_score_meter = AverageMeter()


def train():
    #Setup logging
    logger = setup_logger(logger_name, log_dir)

    start_epoch=1
    best_overall = 0
    stale = 0
    
    if CHECKPOINT is not None:
        if os.path.exists(CHECKPOINT):
            checkpoint = torch.load(CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_overall = checkpoint['best_overall']
            print(f"Resume training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found, start training from epoch 1")
                
    #Logging hyperparameters
    logging_hyperparameters(logger)

    for epoch in range(start_epoch, 1+MAX_EPOCHS):
        start_time = time.time()
        #Train
        model.train()

        #Reset meters
        #Common meter
        train_loss_meter.reset()
        val_loss_meter.reset()
        overall_meter.reset()

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

        logger.info("Start training")
        for batch_idx, (image, mask, label) in enumerate(train_loader):
            n = image.shape[0]
            optimizer.zero_grad()
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            label = label.to(DEVICE)

            #Forward
            output_mask, output_classification = model(image)

            #Cal loss
            loss_segmentation = dice_loss(output_mask, mask)
            loss_classification = focal_loss(output_classification, label, alpha=0.25, gamma=2,reduction='mean')
            train_loss = ALPHA*loss_segmentation + (1 - ALPHA)*loss_classification
            
            train_loss.backward()
            optimizer.step()
            train_loss_meter.update(train_loss.item(), n)
            seg_train_loss_meter.update(loss_segmentation.item(), n)
            cla_train_loss_meter.update(loss_classification.item(), n)
            if batch_idx % 10 == 0:
                logger.info(f"Epoch[{epoch}] Iteration[{batch_idx}/{len(train_loader)}] Loss: {train_loss:.3f}")

        end_time = time.time()
        logger.info(f"Training Result: Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss_meter.avg:.3f}  Segmentation loss: {seg_train_loss_meter.avg:.3f} Classification loss: {cla_train_loss_meter.avg:.3f} Time epoch: {end_time-start_time:.3f}s")

        #Valid
        model.eval()
        with torch.no_grad():
            for batch_idx, (image, mask, label) in enumerate(val_loader):
                n = image.shape[0]
                image = image.to(DEVICE)
                mask = mask.to(DEVICE)
                label = label.to(DEVICE)

                #Forward
                output_mask, output_classification = model(image)

                #Cal loss
                loss_segmentation = dice_loss(output_mask, mask)
                loss_classification = focal_loss(output_classification, label, alpha=0.25, gamma=2,reduction='mean')
                val_loss = ALPHA*loss_segmentation + (1 - ALPHA)*loss_classification


                #Calculate metrics
                #Segmentation: iou, dice, p, r, f1

                mask = F.sigmoid(mask).round().long()
                tp, fp, fn, tn = smp.metrics.get_stats(output_mask, mask, mode='binary', threshold=0.5)


                seg_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

                seg_dice_score = torch.mean((2*tp.sum(0)/(2*tp.sum(0) + fp.sum(0) + fn.sum(0) + 1e-5)))
                seg_precision_score = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
                seg_recall_score = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
                seg_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")


                #Classification: acc, p, r, f1
                label = label.detach().cpu().numpy()
                output_classification = output_classification.argmax(1).detach().cpu().numpy()

                cla_acc = accuracy_score(label, output_classification)
                cla_precision_score = precision_score(label, output_classification, average='macro', zero_division=0)
                cla_recall_score = recall_score(label, output_classification, average='macro', zero_division=0)
                cla_f1_score = f1_score(label, output_classification, average='macro')
                

                #Update meters
                val_loss_meter.update(val_loss.item(), n)

                #Segmentation
                seg_val_loss_meter.update(loss_segmentation.item(), n)
                seg_iou_meter.update(seg_iou_score.item(), n)
                seg_dice_meter.update(seg_dice_score.item(), n)
                seg_precision_meter.update(seg_precision_score.item(), n)
                seg_recall_meter.update(seg_recall_score.item(), n)
                seg_f1_score_meter.update(seg_f1_score.item(), n)

                #Classification
                cla_val_loss_meter.update(loss_classification.item(), n)
                cla_acc_meter.update(cla_acc.item(),n)
                cla_precision_meter.update(cla_precision_score.item(), n)
                cla_recall_meter.update(cla_recall_score.item(), n)
                cla_f1_score_meter.update(cla_f1_score.item(), n)

                #Common
                overall_score = ((seg_iou_score + seg_dice_score + seg_f1_score)/3 + cla_f1_score)/2
                overall_meter.update(overall_score.item(), n)

        logger.info(f"Validation Result: Loss: {val_loss_meter.avg:.3f}, Segmentation loss: {seg_val_loss_meter.avg:.3f} Classification loss: {cla_val_loss_meter.avg:.3f} Overal Score: {overall_meter.avg:.3f}")
        logger.info(f"Classification: Accuracy: {cla_acc_meter.avg:.3f}, F1-Score: {cla_f1_score_meter.avg:.3f}, Precision: {cla_precision_meter.avg:.3f}, Recall: {cla_recall_meter.avg:.3f}")
        logger.info(f"Segmentation: IoU: {seg_iou_meter.avg:.3f} Dice: {seg_dice_meter.avg:.3f}, F1-score: {seg_f1_score_meter.avg:.3f}, Precision: {seg_precision_meter.avg:.3f}, Recall: {seg_recall_meter.avg:.3f}")
        #Save best model
        to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_overall': best_overall
        }
        if overall_meter.avg > best_overall: # best base on IoU score
            logger.info(f"Best model found at epoch {epoch}, saving model")
            torch.save(to_save, os.path.join(weight_dir,f"best_{epoch}_{INPUT_SIZE[0]}_BS={BATCH_SIZE}_overal={overall_meter.avg:.3f}.pth"))
            best_overall = overall_meter.avg
            stale = 0
        else:
            stale += 1
            if stale > 300:
                logger.info(f"No improvement {300} consecutive epochs, early stopping")
                break
        if epoch % SAVE_INTERVAL == 0 or epoch == MAX_EPOCHS:
            logger.info(f"Save model at epoch {epoch}, saving model")

            torch.save(to_save, os.path.join(weight_dir,f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    train()