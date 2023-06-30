import sys
sys.path.append("../dataset/")

import os
os.environ["tf_gpu_allocator"]="cuda_malloc_async"

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import datetime
import glob
import json
import matplotlib.pyplot as plt
import nrrd
import numpy as np
# import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
fix_gpu()


import keras_unet_collection

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import UltraSoundImages
from keras_unet_collection import models
from keras_unet_collection import losses
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from utils import *
from tensorflow.keras import backend as K 




class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="dice_coefficient", **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.union.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        return (2.0 * self.intersection + 1e-5) / (self.union + 1e-5)

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)
        
        
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




parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default='unet_2d', help="U-Net model type")
parser.add_argument("-is", "--input_size", default=256, type=int, help="Input size")
parser.add_argument("-ds", "--dataset_type", default='samsung', help="Choose dataset - Samsung, GE or mix")
parser.add_argument("-de", "--dependency", default=None, help="Continue training on pretrained model")
parser.add_argument("-b", "--batch_size", default=4, type=int, help="Training batch size")
parser.add_argument("-lt", "--loss_type", default='binary_crossentropy', help="Loss function type")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Epochs number")
parser.add_argument("-p", "--patiance", default=10, type=int, help="Training patiance based validation loss")
parser.add_argument("-t", "--test", default=False, type=bool, help="Save model to test directory")
parser.add_argument("-cd", "--custom_directory", default=None, type=str, help="Custom output directory name, instead of date or test")
args = vars(parser.parse_args())


model_type = args['model']
dataset_type = args['dataset_type']
batch_size = args['batch_size']
loss_type = args['loss_type']
epochs = args['epochs']
patiance = args['patiance']
dependency_model = args['dependency']
custom_directory = args['custom_directory']
input_size = args['input_size']


augment = True
image_input_shape = (input_size, input_size, 1)
image_gen_shape = (image_input_shape[0], image_input_shape[1], 1)
# filter_num = [64, 128, 256, 512, 1024]
filter_num = [32, 64, 128]

train_gen, val_gen, test_gen = get_dataset(dataset_type, batch_size, image_gen_shape, augment)

output_path = 'output'
output_path = f'{output_path}/{model_type}/{dataset_type}/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

if custom_directory:
    directory = custom_directory
elif args['test']:
    directory = 'test'
else:
    directory = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

output_path = f'{output_path}{directory}/'
images_path = f'{output_path}/images/' 
meta_path = f'{output_path}/meta/'
model_path = f'{output_path}/model/'
checkpoint_path = f'{output_path}/checkpoint/'
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(images_path)
    os.mkdir(meta_path)
    os.mkdir(model_path)
    os.mkdir(checkpoint_path)

meta_data = {
    'model_type': model_type,
    'epochs': epochs,
    'filter_num': filter_num,
    'dependency': dependency_model,
    'patiance': patiance,
    'dataset': {
        'type': dataset_type,
        'image_shape': image_gen_shape,
        'split_size':{
            'train': len(train_gen.masks),
            'val': len(val_gen.masks),
            'test': len(test_gen.masks),
        },
        'augmentation': augment,
    },
    'loss_function':{
        'type': loss_type,
        'parameters': None,
    }
}

json.dump(meta_data, open(f'{meta_path}meta_data.json', 'w'))

# Callbacks            
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=patiance, # Should be more than 10!
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min'
)

images, masks = val_gen.__getitem__(0, augment=False)
image = images[0]
images_inter_checkpoint = ImagesInterCheckpoint(image, every_n_epoch=10)
callbacks = [early_stopping, images_inter_checkpoint, checkpoint]

# Train with multiple GPUs :)


gpus = tf.config.list_logical_devices('GPU')
if len(gpus) < 3:
    raise('Not enough gpus')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    
#     # Metrics
#     mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)
    metrics = ['accuracy']
    
#     K.clear_session()
    model = get_model(model_type, image_input_shape, filter_num)
    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss_func = get_loss_function(loss_type=loss_type)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=5e-2), loss=loss_func, metrics=metrics)
    # model.compile(optimizer=keras.optimizers.Adam(), loss=loss_func, metrics=metrics)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    
# # Save progress
# model.save(f'{model_path}model')
# history_dict = history.history
# json.dump(history_dict, open(f'{meta_path}history.json', 'w'))
