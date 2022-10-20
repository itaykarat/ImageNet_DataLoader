import os
import CODE.constants as constants
from CODE.data_tranformations import data_transformations
from CODE.main import generate_dataloader
from CODE.display_data_set import display_data_set
from CODE.pre_processing_technical import pre_processing_technical
from CODE.run_model import run_model


"""Flag to display{True/False} the images"""
display: bool = False

if __name__ == '__main__':
    """------------------------------------------ validation data in csv form --------------------------------------"""
    val_data = pre_processing_technical().to_csv()
    print(val_data.head())  # Columns: {file,class,x_coord,y_coord,height,width}

    """------------------------------------------ technical pre-processing ------------------------------------------"""
    # Create separate validation sub-folders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(constants.VALID_DIR, 'images')
    # Open and read val annotations text file
    fp = open(os.path.join(constants.VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Display first 10 entries of resulting val_img_dict dictionary
    for k,v in list(val_img_dict.items())[:10]:
        print(f'class_id: {v} , image_id: {k}')


    # Create subfolders (if not present) for validation images based on label ,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    fp = open(os.path.join(constants.DATA_DIR, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()

    # Display first 20 entries of resulting dictionary
    # print({k: class_to_name_dict[k] for k in list(class_to_name_dict)[:20]})
    for k,v in list(class_to_name_dict.items())[:20]:
        print(f'class: {v}, image_num: {k}')

    # Define transformation sequence for image pre-processing
    # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
    # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225])
    """------------------------------------------ Image transformations ------------------------------------------"""
    preprocess_transform = data_transformations().preprocess_transform()
    preprocess_transform_pretrain = data_transformations().preprocess_transform()

    """-------------------------- Data loaders for train test validation ------------------------------------------"""
    """----- batch of training images -----"""
    train_loader = generate_dataloader(constants.TRAIN_DIR, "train",
                                       transform=preprocess_transform)
    if(display):
        # Display batch of training set images
        display_data_set().show_batch(train_loader)


    """----- batch pre trained normalized images -----"""
    # Create train loader for pre-trained models (normalized based on specific requirements)
    train_loader_pretrain = generate_dataloader(constants.TRAIN_DIR, "train",
                                                transform=preprocess_transform_pretrain)
    if(display):
        # Display batch of pre-train normalized images
        display_data_set().show_batch(train_loader_pretrain)
        # Create dataloaders for validation data (depending if model is pretrained)

    """----- batch of validation set -----"""
    val_loader = generate_dataloader(val_img_dir, "val",
                                     transform=preprocess_transform)
    val_loader_pretrain = generate_dataloader(val_img_dir, "val",
                                              transform=preprocess_transform_pretrain)
    if(display):
        # Display batch of validation images
        display_data_set().show_batch(val_loader)



    """----------- PRINT { features & labels }"""
    features,labels = constants.features_labels_split(train_loader_pretrain)
    print(f'\n\nThose are the features\n {features}')
    print(f'\n\nThose are the labels\n {labels}')



    """----------- RUN THE PRE-TRAINED MODEL ON THE FEATURES WITHOUT LAST LAYER """
    output = run_model().RUN_RESNET18(features=features)


