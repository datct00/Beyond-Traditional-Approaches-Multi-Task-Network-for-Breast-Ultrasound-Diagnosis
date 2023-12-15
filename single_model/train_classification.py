import sys 
sys.path.append("..")
from dataloader import train_loader, val_loader
from hyper import *
from model import classification_model
from lib import torch, os, optim, time, accuracy_score, precision_score, recall_score, f1_score, focal_loss
from utils import AverageMeter, setup_logger, logging_hyperparameters, init_path


#TASK
TASK = "classification"

#Path 
weight_dir, log_dir, logger_name = init_path(TASK)

#Model
model = classification_model().to(DEVICE)

#Loss & Optimizer
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)


#Meters
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
acc_meter = AverageMeter()
precision_meter = AverageMeter()
recall_meter = AverageMeter()
f1_score_meter = AverageMeter()


def train():
    logger = setup_logger(logger_name, log_dir)
    best_f1 = 0
    stale = 0
    start_epoch = 1
    
    if CHECKPOINT is not None:
        if os.path.exists(CHECKPOINT):
            checkpoint = torch.load(CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            logger.info(f"Resume training from epoch {start_epoch}")
        else:
            logger.info(f"Checkpoint not found, start training from epoch 1")

    #Logging hyperparameters
    logging_hyperparameters(logger)


    for epoch in range(start_epoch, 1+MAX_EPOCHS):
        #Start time 
        start_time = time.time()
        #Train
        model.train()
        #Reset meters
        train_loss_meter.reset()
        precision_meter.reset()
        recall_meter.reset()
        f1_score_meter.reset()
        acc_meter.reset()

        logger.info("Start training")
        for batch_idx, (image, _, label) in enumerate(train_loader):
            n = image.shape[0]
            optimizer.zero_grad()
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image) #Logits (batch_size,num_classes)
            #Cal loss
            train_loss = focal_loss(output, label, alpha=0.25, gamma=2, reduction='mean')
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
            for batch_idx, (image, _, label) in enumerate(val_loader):
                n = image.shape[0]
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                output = model(image)
                val_loss = focal_loss(output, label, alpha=0.25, gamma=2,reduction='mean')

                #Calculate metrics
                #P, R and F1
                label = label.detach().cpu().numpy()
                output = output.argmax(1).detach().cpu().numpy()

                p_score = precision_score(label, output, average='macro', zero_division=0)
                r_score = recall_score(label, output, average='macro', zero_division=0)
                _f1_score = f1_score(label, output, average='macro')
                acc = accuracy_score(label, output)

                #Update meters
                val_loss_meter.update(val_loss.item(), n)
                acc_meter.update(acc.item(),n)
                precision_meter.update(p_score.item(), n)
                recall_meter.update(r_score.item(), n)
                f1_score_meter.update(_f1_score.item(), n)

        logger.info(f"Validation Result: Loss: {val_loss_meter.avg:.3f}, Accuracy: {acc_meter.avg:.3f} F1-Score: {f1_score_meter.avg:.3f}, Precision: {precision_meter.avg:.3f}, Recall: {recall_meter.avg:.3f}")

        #Save best model
        to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }
        if f1_score_meter.avg > best_f1: # best base on IoU score
            logger.info(f"Best model found at epoch {epoch}, saving model")
            torch.save(to_save, os.path.join(weight_dir,f"best_{epoch}_{INPUT_SIZE[0]}_BS={BATCH_SIZE}_f1={f1_score_meter.avg:.3f}.pth")) # only save best to prevent output memory exceed error
            best_f1 = f1_score_meter.avg
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