import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

#Training variables
imgSize = 224
batchSize = 32
epochs = 5
trainPath = "./images/asl_alphabet_train/asl_alphabet_train"
testPath = "./images/asl_alphabet_test/asl_alphabet_test"

#Handling gpu usage
# physicalDevices = tf.config.experimental.list_physical_devices('GPU')
# if len(physicalDevices) != 0:
#     tf.config.experimental.set_memory_growth(physicalDevices[0], True)

#Data generators for augmentation and rescaling
trainDataAugment = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                        horizontal_flip=True,
                                        rotation_range=50,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rescale=1./255)
testDataAugment = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input, rescale=1./255)

#Running the data generators
train_dataset = trainDataAugment.flow_from_directory(trainPath,
     shuffle=True,
     classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'],
     target_size=(imgSize, imgSize),
     batch_size=batchSize)
test_dataset = testDataAugment.flow_from_directory(testPath,
     classes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'],
     target_size=(imgSize, imgSize),
     batch_size=batchSize)

#Getting the pretrained mobilnetV2 model from google repo
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
download_model = hub.KerasLayer(url,input_shape=(imgSize,imgSize,3))
mobile = Sequential([
     download_model,
     Dense(29),
     Activation("softmax")
])

#Compiling
mobile.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=['accuracy'])

#training model
mobile.fit(train_dataset, batch_size=batchSize, epochs=epochs)

#evaluating model
# mobile.evaluate(test_dataset)