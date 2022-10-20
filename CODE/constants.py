import os

DATA_DIR = '../tiny-imagenet-200/tiny-imagenet-200'  # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val')

batch_size = 64
image_size = 224


def features_labels_split(train_loader_pretrain):
    data_iter = iter(train_loader_pretrain)
    data = data_iter.next()
    features,labels = data
    return features,labels