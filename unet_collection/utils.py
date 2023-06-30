import sys
sys.path.append("../dataset/")
sys.path.append("TransUnet/")

# import models.transunet as transunet
# import experiments.config as conf

import glob

from dataset import UltraSoundImages
from keras_unet_collection import models, losses

import tensorflow as tf


def get_model(model_type, image_input_shape, filter_num):
    if model_type == 'unet_2d':
        model = models.unet_2d(image_input_shape, filter_num, n_labels=1, output_activation='Sigmoid')
    elif model_type == 'u2net_2d':
        model = models.u2net_2d(image_input_shape, filter_num_down=filter_num, n_labels=1, output_activation='Sigmoid')
    elif model_type == 'unet_3plus_2d':
        model = models.unet_3plus_2d(image_input_shape, n_labels=1, filter_num_down=filter_num, output_activation='Sigmoid')
    elif model_type == 'transunet_2d':
        # model = models.transunet_2d((256, 256, 1), filter_num=filter_num, n_labels=1, output_activation='Sigmoid')
        # model = models.transunet_2d(image_input_shape, filter_num=filter_num, n_labels=1, output_activation='Sigmoid')
        model = models.transunet_2d((320, 320, 1), filter_num=filter_num, n_labels=1, stack_num_down=1, stack_num_up=1,
                                embed_dim=256, num_mlp=1536, num_heads=6, num_transformer=6,
                                activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')
        
        # Config has to be loaded from separate file
        # config = conf.get_transunet()
        # model = transunet.TransUnet(config).model

    elif model_type == 'swin_unet_2d':
        model = models.swin_unet_2d(image_input_shape, filter_num_begin=64, n_labels=1, depth=3, stack_num_down=2, stack_num_up=2, 
                                    num_heads=1, num_mlp=3072, output_activation='Sigmoid')
    return model


def get_model_from_path(path_to_model):
    model = None
    return model


def get_loss_function(loss_type):
    if loss_type=='custom_focal_tversky':
        def custom_focal_tversky(y_true, y_pred, alpha=0.7, gamma=4/3):
            return losses.focal_tversky(y_true, y_pred, alpha=alpha, gamma=gamma)
        return custom_focal_tversky
    
    if loss_type=='dice':
        return losses.dice
    
    if loss_type=='iou_box':
        return losses.iou_box
    
    if loss_type=='binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy
    return None


def get_dataset(dataset_type, batch_size, image_gen_shape, augment, blacklist=None):
  
    # Prepare data generator
    if dataset_type == 'ge':    
        raw_dir = "../dataUSGthyroid/GE_processed"
        blacklist = ['2061093', '1909699', '2026988', '2052396', '2051655', 
                     '2056390', '176349', '1645263', '2060219', '65544']
    elif dataset_type == 'samsung':
        raw_dir = "../dataUSGthyroid/samsung_processed"
        blacklist = ['2089146', '2110713', '2096868', '441540', '2090807', 
                     '2090038', '2091948', '935892', '2058398', '2096659']
    else:
        raise Exception('Sorry, there is no dataset type called: ' + dataset_type)
        
    raw_masks = raw_dir + "/masks"
    raw_images = raw_dir + "/images"
    raw_images_paths = sorted(glob.glob(raw_images + '**/*', recursive=True))[:150]
    raw_masks_paths = sorted(glob.glob(raw_masks + '**/*', recursive=True))[:150]
    
    if blacklist:
        raw_images_paths = _filter_paths(raw_images_paths, blacklist)
        raw_masks_paths = _filter_paths(raw_masks_paths, blacklist)
    
    
    
    TEST_LEN = 16
    # VAL_LEN = 10*16
    VAL_LEN = 3*16
    # VAL_LEN = len(raw_images_paths) // 4
    TRAIN_LEN = len(raw_images_paths) - VAL_LEN - TEST_LEN
        
    train_images = raw_images_paths[:TRAIN_LEN]
    validation_images = raw_images_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
    test_images =  raw_images_paths[-TEST_LEN:]

    train_masks = raw_masks_paths[:TRAIN_LEN]
    validation_masks = raw_masks_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
    test_masks =  raw_masks_paths[-TEST_LEN:]

    train_gen = UltraSoundImages(batch_size, train_images, train_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    val_gen = UltraSoundImages(batch_size, validation_images, validation_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    test_gen = UltraSoundImages(batch_size, test_images, test_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    
    return train_gen, val_gen, test_gen


def _filter_paths(paths, blacklist):
    new_paths = []
    for path in paths:
        file_id = path.split('/')[-1].split('_')[0]
        if file_id not in blacklist:
            new_paths.append(path)
    return new_paths
            
            