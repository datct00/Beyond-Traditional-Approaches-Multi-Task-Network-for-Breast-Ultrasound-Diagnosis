import sys 
sys.path.append("..")
from dataloader import train_loader, val_loader
from hyper import *
from model import segmentation_model
from lib import time, torch, os, smp, optim, F
from utils import AverageMeter, setup_logger, logging_hyperparameters, init_path

#TASK
TASK = "segmentation"

#Path 
weight_dir, log_dir, logger_name = init_path(TASK)


#Model
model = segmentation_model().to(DEVICE)

#Loss & Optimizer
model = model.to(DEVICE)
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True) 
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)


#Meters
overall_meter = AverageMeter()
iou_meter = AverageMeter()
dice_meter = AverageMeter()
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
precision_meter = AverageMeter()
recall_meter = AverageMeter()
f1_score_meter = AverageMeter()

def train():
    logger = setup_logger(logger_name, log_dir)
    stale = 0
    best_overall = 0
    start_epoch = 1
    
    if CHECKPOINT is not None:
        if os.path.exists(CHECKPOINT):
            checkpoint = torch.load(CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_overall = checkpoint['best_overall']
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            logger.info(f"Checkpoint not found, start training from epoch 1")
    #Logging hyperparameters
    logging_hyperparameters(logger)

    for epoch in range(start_epoch, 1+MAX_EPOCHS):
        start_time = time.time()
        #Train
        model.train() 
        #Reset meters 
        overall_meter.reset()
        train_loss_meter.reset()
        val_loss_meter.reset()

        iou_meter.reset()
        dice_meter.reset()
        precision_meter.reset()
        recall_meter.reset()
        f1_score_meter.reset()
        
        logger.info("Start training")
        for batch_idx, (image, mask, _) in enumerate(train_loader):
            n = image.shape[0]
            optimizer.zero_grad()
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            output = model(image) #Logits 
            #Cal loss
            train_loss = dice_loss(output, mask) 
            train_loss.backward() 
            optimizer.step()

            train_loss_meter.update(train_loss.item(),n)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch[{epoch}] Iteration[{batch_idx}/{len(train_loader)}] Loss: {train_loss:.3f}")
        end_time = time.time()
        logger.info(f"Training Result: Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss_meter.avg:.3f}, Time epoch: {end_time-start_time:.3f}s")

        #Valid
        model.eval() 
        with torch.no_grad():
            for batch_idx, (image, mask, _) in enumerate(val_loader): 
                n = image.shape[0]
                image = image.to(DEVICE)
                mask = mask.to(DEVICE)

                output = model(image)
                val_loss = dice_loss(output, mask)

                # #Calculate metrics
                mask = F.sigmoid(mask).round().long()
                tp, fp, fn, tn = smp.metrics.get_stats(output, mask, mode='binary', threshold=0.5)


                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

                dice_score = torch.mean((2*tp.sum(0)/(2*tp.sum(0) + fp.sum(0) + fn.sum(0) + 1e-5)))
                precision_score = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
                recall_score = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")


                #Update meters 
                val_loss_meter.update(val_loss.item(), n)

                iou_meter.update(iou_score.item(), n)
                dice_meter.update(dice_score.item(), n)
                precision_meter.update(precision_score.item(), n)
                recall_meter.update(recall_score.item(), n)
                f1_score_meter.update(f1_score.item(), n)

                #Overall score 
                overall_score = (iou_score + dice_score + f1_score)/3
                overall_meter.update(overall_score.item(), n)

        logger.info(f"Validation Result: Dice Loss: {val_loss_meter.avg:.3f}, IoU: {iou_meter.avg:.3f}, Dice Score: {dice_meter.avg:.3f}, F1-Score: {f1_score_meter.avg:.3f}, Average Score: {overall_meter.avg:.3f}")

        #Save best model 
        to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_overall': best_overall,
            }
        if overall_meter.avg > best_overall: # best base on IoU score
            logger.info(f"Best model found at epoch {epoch}, saving model")
            
            torch.save(to_save, os.path.join(weight_dir,f"best_{epoch}_{INPUT_SIZE[0]}_BS={BATCH_SIZE}_average={overall_meter.avg:.3f}.pth"))
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