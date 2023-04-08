import sys
sys.path.append("../dataset/")

import datetime
import glob
import json
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import tensorflow as tf
import keras_unet_collection

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import UltraSoundImages
from keras_unet_collection import models
from keras_unet_collection import losses
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from utils import plot_history
# from losses import *


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default='unet_2d', help="U-Net model type")
parser.add_argument("-b", "--batch_size", default=4, type=int, help="Training batch size")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Epochs number")
parser.add_argument("-p", "--patiance", default=10, type=int, help="Training patiance based validation loss")
args = vars(parser.parse_args())

# Smaller batch size makes it easier to get out of the local minimum. 


model_type = args['model']
batch_size = args['batch_size']
epochs = args['epochs']
patiance = args['patiance']
image_input_shape = (512, 512, 1)
filter_num = [64, 128, 256, 512]

def get_model(model_type):
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


def get_optimzer(opt):
    if opt=='adam':
        return None
    
    

# output/
#    model_type/
#       /date
#          /model
#          /images
#          /meta
#             history.json
#             other_data.txt #TODO

output_path = 'output'
date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
output_path = f'{output_path}/{model_type}/{date}/'
images_path = f'{output_path}/images/' 
meta_path = f'{output_path}/meta/'
model_path = f'{output_path}/model/'

os.mkdir(output_path)
os.mkdir(images_path)
os.mkdir(meta_path)
os.mkdir(model_path)

# Prepare data generator
raw_dir = "../RawUSGimagesNRRD"
raw_images = raw_dir + "/data_output"
raw_masks = raw_dir + "/masks"

raw_images_paths = sorted(glob.glob(raw_images + '**/*', recursive=True))
raw_masks_paths = sorted(glob.glob(raw_masks + '**/*', recursive=True))

TEST_LEN = 5
VAL_LEN = 60
TRAIN_LEN = len(raw_images_paths) - VAL_LEN - TEST_LEN

train_images = raw_images_paths[:TRAIN_LEN]
validation_images = raw_images_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
test_images =  raw_images_paths[-TEST_LEN:]

train_masks = raw_masks_paths[:TRAIN_LEN]
validation_masks = raw_masks_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
test_masks =  raw_masks_paths[-TEST_LEN:]

train_gen = UltraSoundImages(batch_size, train_images, train_masks, size=(512,512))
val_gen = UltraSoundImages(batch_size, validation_images, validation_masks, size=(512,512))
test_gen = UltraSoundImages(batch_size, test_images, test_masks, size=(512,512))


class MetaData:
    
    def __init__(self):
        pass
    
    
    def add(self):
        pass
    
    
    def save(self):
        pass
    
    
def save_model_summary():
    pass


# Callbacks
class ImagesInterCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, input_image, every_n_epoch=10):
        super(ImagesInterCheckpoint, self).__init__()
        self.every_n_epoch = every_n_epoch
        self.input_image = input_image
        save_img(f'{images_path}input_image.png', self.input_image)
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n_epoch == 0:
            output_mask = self.model.predict(np.expand_dims(self.input_image, axis=0))[0]
            output_mask = (output_mask*255).astype('int')
            save_img(f'{images_path}pred_{epoch}.png', output_mask)
            
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=patiance, # Should be more than 10!
    restore_best_weights=True
)

images, masks = val_gen.__getitem__(0)
image = images[0]
images_inter_checkpoint = ImagesInterCheckpoint(image, every_n_epoch=10)
callbacks = [early_stopping, images_inter_checkpoint]

def cutom_focal_tversky(y_true, y_pred, alpha=0.7, gamma=4/3):
    
    return losses.focal_tversky(y_true, y_pred, alpha=alpha, gamma=gamma)

# Train with multiple GPUs :)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model = get_model(model_type)
    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # keras_unet_collection.losses
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=5e-2), loss=cutom_focal_tversky, metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    
    # Switch loss function mid training
    
# Save progress
model.save(f'{model_path}model')
history_dict = history.history
json.dump(history_dict, open(f'{meta_path}history.json', 'w'))
