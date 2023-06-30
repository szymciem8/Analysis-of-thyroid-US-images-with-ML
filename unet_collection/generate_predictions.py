import tensorflow as tf
from keras_unet_collection import losses
import sys
sys.path.append("../dataset/")
from dataset import UltraSoundImages
import multiprocessing
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import pickle

def get_dataset(dataset_type, batch_size, image_gen_shape, augment, just_test=False):
  
    # Prepare data generator
    if dataset_type == 'ge':    
        raw_dir = "../dataUSGthyroid/GE_processed"
    elif dataset_type == 'samsung':
        raw_dir = "../dataUSGthyroid/samsung_processed"
    else:
        raise Exception('Sorry, there is no dataset type called: ' + dataset_type)
        
    raw_masks = raw_dir + "/masks"
    raw_images = raw_dir + "/images"
    raw_images_paths = sorted(glob.glob(raw_images + '**/*', recursive=True))
    raw_masks_paths = sorted(glob.glob(raw_masks + '**/*', recursive=True))
    
    if not just_test:
        test_images =  raw_images_paths
        test_masks =  raw_masks_paths
    else:
        TEST_LEN = 5
        VAL_LEN = 40
        TRAIN_LEN = len(raw_images_paths) - VAL_LEN - TEST_LEN

        test_images =  raw_images_paths[-TEST_LEN:]
        test_masks =  raw_masks_paths[-TEST_LEN:]

    test_gen = UltraSoundImages(batch_size, test_images, test_masks, size=image_gen_shape, dataset_type=dataset_type, augment=augment)
    
    return test_gen, raw_images_paths


def custom_focal_tversky(y_true, y_pred, alpha=0.8, gamma=4/3):
    return losses.focal_tversky(y_true, y_pred, alpha=alpha, gamma=gamma)

custom_objects = {'custom_focal_tversky': custom_focal_tversky}

def make_predictions(model_path, ds_gen, output_path, output, raw_images_paths):
    images, masks = ds_gen.__getitem__(0)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        output['images'] = images
        output['masks'] = masks
        output['predictions'] = model.predict(images)
        output['images_paths'] = raw_images_paths
    
    with open(output_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-mp", "--model_path", help="Model path")
parser.add_argument("-dt", "--dataset_type", help="Dataset type")
parser.add_argument("-op", "--output_path", help="Output path")
parser.add_argument("-bs", "--batch_size", default=16, help="Batch size")
parser.add_argument("-is", "--image_size", default=256, help="Image size")
parser.add_argument("-au", "--augment", default=False, help="Augment")
parser.add_argument("-jt", "--just_test", type=bool, default=True, help="Just test")
args = vars(parser.parse_args())


model_path = args['model_path']
dataset_type = args['dataset_type']
output_path = args['output_path']
batch_size = args['batch_size']
image_size = args['image_size']
augment = args['augment']
just_test = args['just_test']

if __name__=='__main__':
    output = {}
    ds_gen, raw_images_paths = get_dataset(dataset_type, batch_size, [image_size, image_size], augment, False)
    make_predictions(model_path, ds_gen, output_path, output, raw_images_paths)
    # p = multiprocessing.Process(target=make_predictions, args=(model_path, ds_gen, output_path, output,))
    # p.start()
    # p.join()

