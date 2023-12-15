import sys  
sys.path.append("..")
from lib import smp
from hyper import *
from single_model.model import segmentation_model

RESNET50_ENCODER_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

def multitask_model():
    aux_param=dict(
                    pooling='avg',             # one of 'avg', 'max'
                    dropout=0.5,               # dropout ratio, default is None
                    # activation='sigmoid',      # activation function, default is None
                    classes=CLA_NUM_CLASSES,      # define number of output labels
                )
    model = segmentation_model(aux_param=aux_param)
    return model
if __name__ == "__main__": 
    model = multitask_model()
    print(model)