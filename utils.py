from hyper import *
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    

def calculate_overlap_metrics(pred, gt,eps=1e-5):
    output = pred.view(-1,)
    target = gt.view(-1,).float()

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    # pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = ( tp + eps) / ( tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
#     specificity = (tn + eps) / (tn + fp + eps)

    return iou, dice, precision, recall

def setup_logger(logger_name, output_dir):
    from lib import logging
    import os
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def logging_hyperparameters(logger):
    logger.info("==========Hyperparameters==========")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Architecture: {ARCH}")
    logger.info(f"Encoder: {ENCODER_NAME}")
    logger.info(f"Encoder weight: imagenet")
    logger.info(f"Input size: {INPUT_SIZE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Base learning rate: {BASE_LR}")
    logger.info(f"Max epochs: {MAX_EPOCHS}")
    logger.info(f"Weight decay: {1e-5}")
    logger.info("===================================")


def init_path(task):
    #Task == classification 
    if task == "classification":
        weight_dir = os.path.join(OUTPUT_DIR, task, ENCODER_NAME)
        os.makedirs(weight_dir, exist_ok=True)
        log_dir = weight_dir
        logger_name = f"{task}_{ENCODER_NAME}"
    elif task == "segmentation": 
        weight_dir = os.path.join(OUTPUT_DIR, task, f"{ENCODER_NAME}_{ARCH}")
        os.makedirs(weight_dir, exist_ok=True)
        log_dir = weight_dir
        logger_name = f"{task}_{ENCODER_NAME}_{ARCH}" 
    elif task == "multitask":
        weight_dir = os.path.join(OUTPUT_DIR, f"{ENCODER_NAME}_{ARCH}")
        os.makedirs(weight_dir, exist_ok=True)
        log_dir = weight_dir
        logger_name = f"{task}_{ENCODER_NAME}_{ARCH}"
    return weight_dir, log_dir, logger_name