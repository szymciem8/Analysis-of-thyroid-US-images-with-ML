import sys
sys.path.append("../dataset/")

from dataset import UltraSoundImages
from utils import plot_history

from keras_unet_collection import models
import glob
import nrrd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
import json

raw_dir = "../RawUSGimagesNRRD"
raw_images = raw_dir + "/data_output"
raw_masks = raw_dir + "/masks"

raw_images_paths = sorted(glob.glob(raw_images + '**/*', recursive=True))
raw_masks_paths = sorted(glob.glob(raw_masks + '**/*', recursive=True))

TEST_LEN = 10
VAL_LEN = 60
TRAIN_LEN = len(raw_images_paths) - VAL_LEN - TEST_LEN
BATCH_SIZE = 16

train_images = raw_images_paths[:TRAIN_LEN]
validation_images = raw_images_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
test_images =  raw_images_paths[-TEST_LEN:]

train_masks = raw_masks_paths[:TRAIN_LEN]
validation_masks = raw_masks_paths[-(VAL_LEN+TEST_LEN):-TEST_LEN]
test_masks =  raw_masks_paths[-TEST_LEN:]

train_gen = UltraSoundImages(BATCH_SIZE, train_images, train_masks, size=(512,512))
val_gen = UltraSoundImages(BATCH_SIZE, validation_images, validation_masks, size=(512,512))
test_gen = UltraSoundImages(BATCH_SIZE, test_images, test_masks, size=(512,512))

class ImagesInterCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, input_image, every_n_epoch=10):
        super(ImagesInterCheckpoint, self).__init__()
        self.every_n_epoch = every_n_epoch
        self.input_image = input_image
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n_epoch == 0:
            output_mask = self.model.predict(np.expand_dims(self.input_image, axis=[0,3]))[0]
            output_mask = (output_mask*255).astype('int')
            save_img(f'images/pred_{epoch}.png', output_mask)

images, masks = val_gen.__getitem__(0)
image = images[0]
images_inter_checkpoint = ImagesInterCheckpoint(image, every_n_epoch=10)

#callbacks = [early_stopping, images_inter_checkpoint]

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

with strategy.scope():

	standard_unet = models.unet_2d((512, 512, 1), [64, 128, 256, 512, 1024], n_labels=1, output_activation='Sigmoid')
	standard_unet.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-1), loss='binary_crossentropy', metrics=['accuracy'])
	history = standard_unet.fit(train_gen, validation_data=val_gen, epochs=10)
	standard_unet.save('models/standard_unet')

	history_dict = history.history
	# Save it under the form of a json file
	json.dump(history_dict, open('history_standard_unet.json', 'w'))
