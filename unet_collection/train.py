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
from utils import *


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default='unet_2d', help="U-Net model type")
parser.add_argument("-ds", "--dataset_type", default='samsung', help="Choose dataset - Samsung, GE or mix")
parser.add_argument("-de", "--dependency", default=None, help="Continue training on pretrained model")
parser.add_argument("-b", "--batch_size", default=4, type=int, help="Training batch size")
parser.add_argument("-lt", "--loss_type", default='binary_crossentropy', help="Loss function type")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Epochs number")
parser.add_argument("-p", "--patiance", default=10, type=int, help="Training patiance based validation loss")
parser.add_argument("-t", "--test", default=False, type=bool, help="Save model to test directory")
args = vars(parser.parse_args())


model_type = args['model']
dataset_type = args['dataset_type']
batch_size = args['batch_size']
loss_type = args['loss_type']
epochs = args['epochs']
patiance = args['patiance']
dependency_model = args['dependency']

augment = True
image_input_shape = (256, 256, 1)
image_gen_shape = (image_input_shape[0], image_input_shape[1])
filter_num = [64, 128, 256, 512, 1024]

train_gen, val_gen, test_gen = get_dataset(dataset_type, batch_size, image_gen_shape, augment)

output_path = 'output'
output_path = f'{output_path}/{model_type}/{dataset_type}/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

if args['test']:
    directory = 'test'
else:
    directory = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

output_path = f'{output_path}{directory}/'
images_path = f'{output_path}/images/' 
meta_path = f'{output_path}/meta/'
model_path = f'{output_path}/model/'
if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(images_path)
    os.mkdir(meta_path)
    os.mkdir(model_path)

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

images, masks = val_gen.__getitem__(0, augment=False)
image = images[0]
images_inter_checkpoint = ImagesInterCheckpoint(image, every_n_epoch=10)
callbacks = [early_stopping, images_inter_checkpoint]


# Train with multiple GPUs :)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    # Metrics
    mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)
    metrics = ['accuracy', mean_iou]
    
    model = get_model(model_type, image_input_shape, filter_num)
    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # keras_unet_collection.losses
    loss_func = get_loss_function(loss_type='custom_focal_tversky')
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=5e-2), loss=loss_func, metrics=metrics)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    
# Save progress
model.save(f'{model_path}model')
history_dict = history.history
json.dump(history_dict, open(f'{meta_path}history.json', 'w'))
