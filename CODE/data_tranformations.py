from torchvision import transforms as T
import constants

class data_transformations:
    def __init__(self):
        pass


    def preprocess_transform(self):
        preprocess_transform_pretrain = T.Compose([
            T.Resize(constants.image_size),  # Resize images to 224 x 224
            # T.CenterCrop(224),  # Center crop image
            # T.RandomHorizontalFlip(),
            T.ToTensor(),  # Converting cropped images to tensors
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        return preprocess_transform_pretrain