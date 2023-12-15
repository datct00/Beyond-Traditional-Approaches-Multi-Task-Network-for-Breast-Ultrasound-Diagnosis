import sys 
sys.path.append("..")
from lib import smp, torchvision, torch
from hyper import *


PRETRAINED_WEIGHT_URL = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'tu-wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'efficientnet-b4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth',
}

def segmentation_model(aux_param=None):
    assert ARCH in ['unet', 'unetpp', 'deeplabv3plus', 'fpn'], "Invalid architecture, must be ['unet', 'unetpp', 'deeplabv3plus', 'fpn']"
    assert ENCODER_NAME in ['resnet50', 'resnext50_32x4d', 'tu-wide_resnet50_2', 'efficientnet-b4'], "Invalid encoder name, must be ['resnet50', 'resnext50_32x4d', 'tu-wide_resnet50_2', 'efficientnet-b4']"
    #Params

    params = dict(
        encoder_name = ENCODER_NAME,
        encoder_depth = 5,
        encoder_weights = "imagenet",
        in_channels = IN_CHANNELS,
        classes = SEG_NUM_CLASSES,
        activation = OUTPUT_ACTIVATION,
        aux_params = aux_param
    )
    MODELS = {
        'unet':smp.Unet(**params),
        'unetpp': smp.UnetPlusPlus(**params),
        'deeplabv3plus': smp.DeepLabV3Plus(**params),
        'fpn': smp.FPN(**params),
        
    }
    return MODELS[ARCH]

def classification_model():
    MODELS = {
        'resnet50': torchvision.models.resnet50(weights='DEFAULT'),
        'resnext50_32x4d': torchvision.models.resnext50_32x4d(weights='DEFAULT'),
        'tu-wide_resnet50_2': torchvision.models.wide_resnet50_2(weights='DEFAULT'),
        'efficientnet-b4': torchvision.models.efficientnet_b4(weights=None),
    }
    model = MODELS[ENCODER_NAME]

    # Replace the last layer
    if ENCODER_NAME == "efficientnet-b4":
        state_dict = torch.hub.load_state_dict_from_url(PRETRAINED_WEIGHT_URL[ENCODER_NAME])
        model.load_state_dict(state_dict)
        model.classifier = torch.nn.Linear(1792, CLA_NUM_CLASSES)
    else:
        model.fc = torch.nn.Linear(2048, CLA_NUM_CLASSES)

    return model

class TwoSingleModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.seg_model = segmentation_model()
        self.cla_model = classification_model()

    def forward(self, x): 
        seg_out = self.seg_model(x)
        cla_out = self.cla_model(x)
        return seg_out, cla_out

if __name__ == "__main__":
    model = classification_model()
    x =  torch.randn(1,3,448,448)
    output = model(x)
    print("Model name: ", model.__class__)
    print(output.shape)
