# Breast Ultrasound

## Overview

Nothing.

## Requirements 
- Python >=3.8
- PyTorch >=1.9 
- Torchvision > 0.16
- segmentation-models-pytorch 

## Getting Started
0. Download dataset from [Merged mask BUSI dataset](https://drive.google.com/drive/folders/11_5ikByF8hkQ7lEyxgpmlnli666vLOMA?usp=sharing) and affirm number of images in the dataset by:

```bash
python check_dataset.py
```
If there are any missing, you can download the original dataset from [here](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) then run the merge mask code to generate same our experimental dataset by: 
```bash
python merge_mask_BUSI.py
```
1. Clone the repository
```bash
git clone https://github.com/datct00/breast_ultrasound.git
cd breast_us
```

2. Install the dependencies
```bash
pip install -r requirements.txt
```

3. Adjust `ARCH`, `ENCODER_NAME` and `OUTPUT_DIR` parameter in `hyper.py` file to indicate different architectures and backbones.

4. Train classification
```bash
cd single_model
python train_classification.py
```

5. Train segmentation
```bash
cd single_model
python train_segmentation.py
```


### Acknoweledge 
@misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}