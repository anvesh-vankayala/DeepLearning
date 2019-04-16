from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os
import shutil
import random

img_width, img_height = 150, 150
epochs = 20
batch_size = 20

from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
train_dir_original='/var/www/projects/intw/anvesh/data/train'
target_base_dir = '/var/www/projects/intw/anvesh/data/target'
test_dir_original = '/var/www/projects/intw/anvesh/data/test'

def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')
    classes = []
    for subdir in os.listdir(train_dir_original):
        classes.append(subdir)
    print(classes)

    if os.path.exists(target_base_dir):
        print('required directory structure already exists. learning continues with existing data')
    else:          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        
        for c in classes: 
            os.mkdir(os.path.join(train_dir, c))
            os.mkdir(os.path.join(validation_dir, c))]
        print('created the required directory structure')
        
        shutil.move(test_dir_original, test_dir)
        print('moving of test data to target test directory finished')
        
        for c in classes:
            sudir = os.path.join(train_dir_original, c)
            files = os.listdir(sudir)
            train_files = [os.path.join(sudir, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  

            for t in train:
                shutil.copy2(t, os.path.join(train_dir, c))
            for v in val:
                shutil.copy2(v, os.path.join(validation_dir, c))
        print('moving of input data to train and validation folders finished')

    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in classes:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in classes:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)    

   # nb_test_samples = len(os.listdir(os.path.join(test_dir, os.listdir(test_dir)[0])))
   # print('total test images:', nb_test_samples )
    
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples
	
train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    preapare_full_dataset_for_flow(
                            train_dir_original, 
                            test_dir_original,
                            target_base_dir)
							

base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(3, activation='softmax'))
model = Model(inputs = base_model.input, outputs = top_model(base_model.output))

set_trainable = False
for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print(model.summary())

## Model complilation
model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

##Generators
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

##Call backs
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

##Fit generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, early_stopping])

historydf = pd.DataFrame(history.history, index=history.epoch)
