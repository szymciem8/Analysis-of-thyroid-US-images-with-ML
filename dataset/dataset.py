import glob
import nrrd
import random
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

SAMSUNG_MASK_NORM = 15555.0
GA_MASK_NORM = 255.0


class UltraSoundImages(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, raw_images_paths, raw_masks_paths, dataset_type='ge', augment=True, random_crop=False, size=None):
        self.MAX_SHARPNESS_LEVEL = 5
        self.MIN_CONTRAST_ADJUSTMENT = 0
        self.MAX_CONTRAST_ADJUSTMENT = 2.5
        self.MAX_NOISE_SCALE = 4
        self.ROTATION_RANGE = 15
        
        self.random_crop = random_crop
        self.batch_size = batch_size
        self.augment = augment
        self.images = []
        self.masks = []
        self.dataset_type = dataset_type
        
        if self.dataset_type == 'samsung':
            self.mask_normalizer = SAMSUNG_MASK_NORM
        else:
            self.mask_normalizer = GA_MASK_NORM

        self.miscellaneous_process = [
            self.sharpen,
            self.adjust_contrast,
            self.adjust_brightness,
            self.add_noise,
            self.gaussian_filter,
        ]
        
        # Image and mask have to be stacked
        self.geometric_process = [
            self.zoom,
            self.flip,
            self.rotate, 
            self.shear,
            # self.random_crop,
        ]
        
        
        print('Loading images from NRRD format and resizing')
        for i in range(len(raw_images_paths)):
            image, header = nrrd.read(raw_images_paths[i])
            mask, header = nrrd.read(raw_masks_paths[i])
            
            if size and not self.random_crop:
                image = tf.image.resize_with_pad(image, size[0], size[1])
                mask = tf.image.resize_with_pad(mask, size[0], size[1])
                
            if self.dataset_type=='ge':
                image = np.expand_dims(image[:,:,0], 2)
                mask = np.expand_dims(mask[:,:,0] , 2)
            
            # self.images.append(tf.convert_to_tensor(list(image)) / 255 )
            # self.masks.append(tf.convert_to_tensor(list(mask)) / 255)
            
            self.images.append(image)
            self.masks.append(mask)
        print('Finished loading')

    def __len__(self):
        return len(self.masks) // self.batch_size
    
    def __getitem__(self, idx, augment=True):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        output_images = self.images[i : i + self.batch_size]
        output_masks = self.masks[i : i + self.batch_size]
        
        if self.augment and augment:
            output_images, output_masks = self._augment_batch(output_images, output_masks)
            
            
        output_images, output_masks = np.array(output_images) / 255, np.array(output_masks) / self.mask_normalizer
        
        return tf.convert_to_tensor(output_images, np.float32), tf.convert_to_tensor(output_masks, np.float32)
    
    def show_sample(self):
        images, masks = self.__getitem__(0)

        i = 0
        plt.figure(figsize=(15, 15))
        for image, mask in zip(images, masks):
            plt.subplot(2, 2, i + 1)
            plt.imshow(image)
            plt.imshow(mask, alpha=0.5)
            plt.axis('off')
            i+=1
        plt.show()
        
    def test_augmentation(self, show_mask=True):
        images, masks = self.__getitem__(0)
        
        image, mask = images[0], masks[0]
        org_image = np.array(image, copy=True)
        org_mask = np.array(mask, copy=True)
        aug_image, aug_mask = self._augment(image, mask)
        
        plt.figure(figsize=(15,15))
        plt.subplot(2,1,1)
        plt.imshow(org_image)
        if show_mask:
            plt.imshow(org_mask, alpha=0.5)
        plt.subplot(2,1,2)
        plt.imshow(aug_image)
        if show_mask:
            plt.imshow(aug_mask, alpha=0.5)
        plt.axis('off')
        plt.show()
    
    
    def _augment_batch(self, images, masks):
        for i in range(len(images)):
            images[i], masks[i] = self._augment(images[i], masks[i])
        return images, masks
    
    def _augment(self, image, mask):
        
        # TODO -> Chose random proc from misc
        # And then perform all geo procs
        # for proc in self.miscellaneous_process:
        #     image, mask = proc(image, mask)
            
#         misc_proc = random.choice(self.miscellaneous_process)
#         image, mask = misc_proc(image, mask)
            
        image, mask = np.array(image), np.array(mask)
        stacked = tf.stack([image.squeeze(), mask.squeeze()])
        for proc in self.geometric_process:
            stacked = proc(stacked)
            
        stacked = np.expand_dims(stacked, 3)
        return stacked
        return tf.unstack(stacked, axis=2)
    
    # MIDCELLANEOUS TRANSFORMATIONS
    def sharpen(self, image, mask):
        level = random.random() * self.MAX_SHARPNESS_LEVEL
        sharpened = tfa.image.sharpness(image, level)
        return sharpened, mask
    
    def adjust_contrast(self, image, mask):
        adjusted = tf.image.random_contrast(image, self.MIN_CONTRAST_ADJUSTMENT, self.MAX_CONTRAST_ADJUSTMENT)
        return adjusted, mask
    
    def adjust_brightness(self, image, mask):
        level = random.random() * 0.5 - 0.1
        bright = tf.image.adjust_brightness(image, level)
        return bright, mask
    
    def add_noise(self, image, mask):
        noise = np.random.normal(size=image.shape, scale=self.MAX_NOISE_SCALE)
        return image+noise, mask
    
    def gaussian_filter(self, image, mask):
        filtered = tfa.image.gaussian_filter2d(image)
        return filtered, mask

    
    # GEOMETRIC
    def flip(self, stacked):
        stacked = tf.stack(tf.unstack(stacked, axis=0), axis=2)
        
        stacked = tf.image.random_flip_left_right(stacked)
        stacked = tf.image.random_flip_up_down(stacked)
        
        stacked = tf.stack(tf.unstack(stacked, axis=2))
        return stacked
    
    def rotate(self, stacked):
        rotated = tf.keras.preprocessing.image.random_rotation(
            stacked,
            self.ROTATION_RANGE,
            fill_mode='constant',
        )
        return rotated
    
    def zoom(self, stacked):
        zoomed = tf.keras.preprocessing.image.random_zoom(
            stacked,
            # 1.5,
            [0.8, 1.2],
            fill_mode='constant',
        )
        return zoomed
    
    def shear(self, stacked):
        sheard = tf.keras.preprocessing.image.random_shear(
            stacked,
            10,
            fill_mode='constant',
        )
        return sheard
    
    # Probably not needed anymore, since other methods perform similar task
    def random_crop(self, image, mask):
        stacked_image = tf.stack([image, mask], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, 450, 450, 1])
        cropped_image, cropped_mask = cropped_image
        return cropped_image, cropped_mask
    
    
    
class Dataset:
    
    def __init__(self, kfold=4):
        pass
    
    def get_fold(self, fold_id):
        pass
    
    def _load_dataset(self):
        pass