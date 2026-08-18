# Training Support

This folder contains an optional training workflow for DenseNet121.

## Dataset Format

The dataset directory should contain subfolders for each class:

- normal/
- pneumonia/

## Install Training Dependencies

pip install -r training/requirements-training.txt

## Run Training

python training/train_densenet121.py --dataset-dir path/to/chest_xray_dataset

Optional arguments include:

- --output-dir training/artifacts
- --image-size 224
- --batch-size 16
- --epochs-frozen 10
- --epochs-finetune 10

## Training Outputs

The script generates:

- Best checkpoint model (.keras)
- Final fine-tuned model (.keras)
- Confusion matrix image
- ROC curve image
- Training curves image
- Evaluation metrics JSON
