import sys
sys.path.append("../dataset/")
# sys.path.append("TransUnet/")

# import models.transunet as transunet
# import experiments.config as conf

import glob

from dataset import UltraSoundImages
import random
from keras_unet_collection import models, losses

import tensorflow as tf
import os

SEED = 42


def get_model(model_type, image_input_shape, filter_num):
    if model_type == 'unet_2d':
        model = models.unet_2d((320, 320, 1), [32, 64, 128, 256, 512, 1024], n_labels=1, output_activation='Sigmoid', stack_num_down=3, stack_num_up=3)
    elif model_type == 'u2net_2d':
        model = models.u2net_2d(image_input_shape, filter_num_down=[32, 64, 128, 256, 512], n_labels=1, output_activation='Sigmoid')
    elif model_type == 'unet_3plus_2d':
        model = models.unet_3plus_2d((320, 320, 1), n_labels=1, filter_num_down=[32, 64, 128, 256, 512, 1024], stack_num_down=4, stack_num_up=4, output_activation='Sigmoid')
    elif model_type == 'transunet_2d':
        model = models.transunet_2d((320, 320, 1), filter_num=[32, 64, 128, 256, 512, 1024], n_labels=1, stack_num_down=2, stack_num_up=2, embed_dim=384, num_mlp=1536, num_heads=8, num_transformer=8, activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid', name='transunet')
    elif model_type == 'swin_unet_2d':
        model = models.swin_unet_2d(input_size=(320,320,1), filter_num_begin=64, n_labels=1, depth=5, stack_num_down=2, stack_num_up=2, 
                                    patch_size=(2,2), num_heads=[16, 16, 16, 16, 16], window_size=[8, 8, 8, 4, 2], num_mlp=768, output_activation='Sigmoid')
        
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


def get_dataset(dataset_type, batch_size, image_gen_shape, augment, nfold=None, fold_id=None, blacklist=None):
    
    dirname = os.path.dirname(__file__)
    samsung_directory = os.path.join(dirname, '../dataUSGthyroid/samsung_processed')
    ge_directory = os.path.join(dirname, '../dataUSGthyroid/GE_processed')
    
    samsung_images_paths = sorted(glob.glob(samsung_directory+"/images"+'**/*', recursive=True))
    samsung_masks_paths = sorted(glob.glob(samsung_directory+"/masks"+'**/*', recursive=True))
    
    ge_images_paths = sorted(glob.glob(ge_directory+"/images"+'**/*', recursive=True))
    ge_masks_paths = sorted(glob.glob(ge_directory+"/masks"+'**/*', recursive=True))
    
    samsung_blacklist = [2089146, 2110713, 2096868, 441540, 2090807, 
                         2090038, 2091948, 935892, 2058398, 2096659]
    
    ge_blacklist = [2052396,  258419,   96874, 1869459,  176349, 2057199, 2056390,
                     65544,  129178, 1898379,  278956,   58865, 2060219,  984089,
                   1294777, 2058490,   56112, 1465881, 1645263, 1894479, 1990099,
                   2064524, 2056143,  122356, 2056594,  367643,  441046, 2056262,
                   1800412,  408920, 2044607, 2058919, 2057976, 2054476,   72899,
                   1232907, 2057729, 2052805, 2049987, 2058083, 1821959, 2039592,
                   2055940, 2049948, 1514002,  430336, 2058336, 2047914, 1987859,
                    962421, 2058232, 2067865,  178682, 2041545,  570138, 1837579,
                   1792356, 1777697,  112232, 2050494,    5450, 2066129, 2067891,
                   1962159, 2058436, 2054920,   19254, 2042518, 2058860,  102497,
                     91963,  333089, 2058455,  312222, 2062047,  294453,  102701,
                   2056173, 2058964, 2058198, 2040985, 2057821, 2059885, 2057144,
                   1982360, 2008804,  147452, 1021419, 2055255, 1496522,  451543,
                   1907420,   90321, 2063236, 1108818,  181555,  260856, 2057879,
                   2055091, 2052986, 2057487, 2045037, 2059137, 2014206, 1377277,
                   2057571, 2057492, 1782753, 2057712,   63919,   26417,   47541,
                   2044639,  551887, 2057834, 1049978, 1964701,  330193,  266298,
                   2047732,   62201, 2056549,  116261,  285541,   94531, 2063459,
                   2048695,  433001,  364011,  644500, 1861159, 2055603,  787850,
                   1443201, 2059268,   64233, 2058493,   68565,   94443, 2060236,
                   2046446, 2057718,   70279,  172572, 2055787,  613693,  711736,
                   1826726, 1814795, 2045927,  108849, 2059940, 2061524,    5226,
                   1244313, 2057039, 2055248,  983846, 2057623,  480077,   27478,
                   2055404, 2048935, 2045932,  931471, 2057795, 1489281, 2044710]

    samsung_images_paths = _shuffle_list(_filter_paths(samsung_images_paths, samsung_blacklist), SEED)
    samsung_masks_paths = _shuffle_list(_filter_paths(samsung_masks_paths, samsung_blacklist), SEED)
    
    ge_images_paths = _shuffle_list(_filter_paths(ge_images_paths, ge_blacklist), SEED)
    ge_masks_paths = _shuffle_list(_filter_paths(ge_masks_paths, ge_blacklist), SEED)
    
    if len(samsung_images_paths) > len(ge_images_paths):
        max_n_of_files = len(ge_images_paths)
        samsung_images_paths = samsung_images_paths[:max_n_of_files]
        samsung_masks_paths = samsung_masks_paths[:max_n_of_files]
        
    if dataset_type == 'samsung':
        images_paths, masks_paths = samsung_images_paths, samsung_masks_paths
    elif dataset_type == 'ge':
        images_paths, masks_paths = ge_images_paths, ge_masks_paths
    elif dataset_type == 'mix':
        s_train_images, s_train_masks, s_val_images, s_val_masks, s_test_images, s_test_masks = _split_paths(samsung_images_paths, samsung_masks_paths, nfold, fold_id, test_size=64)
        g_train_images, g_train_masks, g_val_images, g_val_masks, g_test_images, g_test_masks = _split_paths(ge_images_paths, ge_masks_paths, nfold, fold_id, test_size=64)
        
        size = len(ge_images_paths) // 2
        
        # Half Samsung, Half GE and sorted again
        train_images = _shuffle_list(s_train_images[:size] + g_train_images[:size], SEED)
        train_masks = _shuffle_list(s_train_masks[:size] + g_train_masks[:size], SEED)
        
        val_images = _shuffle_list(s_val_images[:size] + g_val_images[:size], SEED)
        val_masks = _shuffle_list(s_val_masks[:size] + g_val_masks[:size], SEED)
        
        test_images = _shuffle_list(s_test_images[:size] + g_test_images[:size], SEED)
        test_masks = _shuffle_list(s_test_masks[:size] + g_test_masks[:size], SEED)
        
    else:
        raise Exception('Sorry, there is no dataset type called: ' + dataset_type)
    # print(f'Dataset length: {len(masks_paths)}')

    if dataset_type != 'mix':
        train_images, train_masks, val_images, val_masks, test_images, test_masks = _split_paths(images_paths, masks_paths, nfold, fold_id, test_size=64)

    train_gen = UltraSoundImages(batch_size, train_images, train_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    val_gen = UltraSoundImages(batch_size, val_images, val_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    test_gen = UltraSoundImages(batch_size, test_images, test_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    
    return train_gen, val_gen, test_gen


def _shuffle_list(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst


def _split_paths(images, masks, n_fold, fold_id, test_size):
    total_samples = len(images)
    test_end_index = test_size

    test_images = images[:test_end_index]
    test_masks = masks[:test_end_index]

    train_images = images[test_end_index:]
    train_masks = masks[test_end_index:]

    total_train_samples = len(train_images)
    fold_size = total_train_samples // n_fold
    start_index = fold_id * fold_size
    end_index = start_index + fold_size

    val_images = train_images[start_index:end_index]
    val_masks = train_masks[start_index:end_index]

    train_images_fold = train_images[:start_index] + train_images[end_index:]
    train_masks_fold = train_masks[:start_index] + train_masks[end_index:]

    return train_images_fold, train_masks_fold, val_images, val_masks, test_images, test_masks



def _filter_paths(paths, blacklist):
    new_paths = []
    for path in paths:
        file_id = path.split('/')[-1].split('_')[0]
        if int(file_id) not in blacklist:
            new_paths.append(path)
    return new_paths