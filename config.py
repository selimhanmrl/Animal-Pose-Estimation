import torch
# constant paths
ROOT_PATH = 'labeled-data'
#OUTPUT_PATH = 'Demo/Outputs_Resnet18_500/'
# learning parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 0.1
# show dataset keypoint plot
SHOW_DATASET_PLOT = True

