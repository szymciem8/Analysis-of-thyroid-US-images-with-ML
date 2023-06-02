import sys
sys.path.append("../dataset/")

import glob

from dataset import UltraSoundImages
from keras_unet_collection import models, losses


def get_model(model_type, image_input_shape, filter_num):
    if model_type == 'unet_2d':
        model = models.unet_2d(image_input_shape, filter_num, n_labels=1, output_activation='Sigmoid')
    elif model_type == 'u2net_2d':
        model = models.u2net_2d(image_input_shape, filter_num_down=filter_num, n_labels=1, output_activation='Sigmoid')
    elif model_type == 'unet_3plus_2d':
        model = models.unet_3plus_2d(image_input_shape, n_labels=1, filter_num_down=filter_num, output_activation='Sigmoid')
    elif model_type == 'transunet_2d':
        # model = models.transunet_2d(image_input_shape, filter_num=filter_num, n_labels=1, output_activation='Sigmoid')
        model = models.transunet_2d((512, 512, 1), filter_num=[64, 128, 256, 512], n_labels=1, stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=3072, num_heads=1, num_transformer=1,
                                activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')
    elif model_type == 'swin_unet_2d':
        model = models.swin_unet_2d(image_input_shape, filter_num_begin=64, n_labels=1, depth=3, output_activation='Sigmoid')
    return model


def get_model_from_path(path_to_model):
    model = None
    return model


def get_loss_function(loss_type):
    if loss_type=='custom_focal_tversky':
        def custom_focal_tversky(y_true, y_pred, alpha=0.8, gamma=4/3):
            return losses.focal_tversky(y_true, y_pred, alpha=alpha, gamma=gamma)
        return custom_focal_tversky
    return None


def get_dataset(dataset_type, batch_size, image_gen_shape, augment):
  
    # Prepare data generator
    if dataset_type == 'ge':    
        raw_dir = "../RawUSGimagesNRRD"
        raw_images = raw_dir + "/data_output"
    elif dataset_type == 'samsung':
        raw_dir = "../dataUSGthyroid/samsung_processed"
        raw_images = raw_dir + "/images"
    else:
        raise Exception('Sorry, there is no dataset type called: ' + dataset_type)
        
    raw_masks = raw_dir + "/masks"
    raw_images_paths = sorted(glob.glob(raw_images + '**/*', recursive=True))
    raw_masks_paths = sorted(glob.glob(raw_masks + '**/*', recursive=True))
    
    TEST_LEN = 5
    VAL_LEN = 40
    TRAIN_LEN = len(raw_images_paths) - VAL_LEN - TEST_LEN
        
    train_images = raw_images_paths[:TRAIN_LEN]
    validation_images = raw_images_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
    test_images =  raw_images_paths[-TEST_LEN:]

    train_masks = raw_masks_paths[:TRAIN_LEN]
    validation_masks = raw_masks_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
    test_masks =  raw_masks_paths[-TEST_LEN:]

    train_gen = UltraSoundImages(batch_size, train_images, train_masks, size=image_gen_shape, augment=augment)
    val_gen = UltraSoundImages(batch_size, validation_images, validation_masks, size=image_gen_shape, augment=augment)
    test_gen = UltraSoundImages(batch_size, test_images, test_masks, size=image_gen_shape, augment=augment)
    
    
    return train_gen, val_gen, test_gen