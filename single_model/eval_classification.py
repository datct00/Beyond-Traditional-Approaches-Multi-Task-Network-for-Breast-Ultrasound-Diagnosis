#Evaluation
import sys 
sys.path.append("..")
from dataloader import val_loader
from hyper import *
from model import classification_model
from lib import torch, Accuracy, Precision, Recall, FBetaScore, tqdm
from utils import AverageMeter

from utils import calculate_overlap_metrics
from sklearn.metrics import precision_score, recall_score, f1_score 


#model 
model = classification_model().to(DEVICE)
model.load_state_dict(torch.load(WEIGHT, map_location=DEVICE)['model_state_dict'])
model.eval()
print("Evaluating...")
progress = tqdm(val_loader, total=int(len(val_loader)))


#Metric: Precision, Recall, F1 score
acc_fn = Accuracy(num_classes=CLA_NUM_CLASSES, task="multiclass").to(DEVICE)
precision_fn = Precision(num_classes=CLA_NUM_CLASSES, task='multiclass', average='macro').to(DEVICE)
recall_fn = Recall(num_classes=CLA_NUM_CLASSES, task='multiclass', average='macro').to(DEVICE)
f1_score_fn = FBetaScore(num_classes=CLA_NUM_CLASSES, task="multiclass", beta=1.0).to(DEVICE)


#Meters
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()
acc_meter = AverageMeter()
precision_meter = AverageMeter()
recall_meter = AverageMeter()
f1_score_meter = AverageMeter()


sk_precision_meter = AverageMeter()
sk_recall_meter = AverageMeter()
sk_f1_meter = AverageMeter()


def eval():
    acc_meter.reset()
    precision_meter.reset()
    recall_meter.reset()
    f1_score_meter.reset()

    sk_precision_meter.reset() 
    sk_recall_meter.reset()
    sk_f1_meter.reset()
    with torch.no_grad():
        for batch_idx, (image, _, label) in enumerate(progress):
            progress.refresh()
            n = image.shape[0]
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            #Calculate metrics
            #P, R and F1
            _precision_score = precision_fn(output, label)
            _recall_score = recall_fn(output, label)
            _f1_score = f1_score_fn(output, label)
            acc = acc_fn(output,label)

            sk_precision_score = precision_score(output.argmax(1).detach().cpu().numpy(), label.detach().cpu().numpy(), average='macro',zero_division=0)
            sk_recall_score = recall_score(output.argmax(1).detach().cpu().numpy(), label.detach().cpu().numpy(), average='macro',zero_division=0)
            sk_f1_score = f1_score(output.argmax(1).detach().cpu().numpy(), label.detach().cpu().numpy(), average='macro',zero_division=0)

            #Update meters
            acc_meter.update(acc.item(),n)
            precision_meter.update(_precision_score.item(), n)
            recall_meter.update(_recall_score.item(), n)
            f1_score_meter.update(_f1_score.item(), n)

            sk_precision_meter.update(sk_precision_score, n)
            sk_recall_meter.update(sk_recall_score, n)
            sk_f1_meter.update(sk_f1_score, n)


    print(f"\nEvaluation Result: Accuracy: {acc_meter.avg:.3f} F1-Score: {f1_score_meter.avg:.3f}, Precision: {precision_meter.avg:.3f}, Recall: {recall_meter.avg:.3f}")
    print(f"\nSKLearn    Result: Accuracy: {acc_meter.avg:.3f} F1-Score: {sk_f1_meter.avg:.3f}, Precision: {sk_precision_meter.avg:.3f}, Recall: {sk_recall_meter.avg:.3f}")

if __name__ == "__main__":
    eval()