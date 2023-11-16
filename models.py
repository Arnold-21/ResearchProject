import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os

#Toggling gpu or cpu use
gpuUse = True
if not gpuUse:
     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Training variables
imgSize = 224
batchSize = 32
epochs = 5
trainPath = "./images/asl_alphabet_train/asl_alphabet_train"
testPath = "./images/asl_alphabet_test/asl_alphabet_test"

#Handling gpu usage by setting memory growth
physicalDevices = tf.config.experimental.list_physical_devices('GPU')
if len(physicalDevices) != 0:
    tf.config.experimental.set_memory_growth(physicalDevices[0], True)

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

#Reading the the experiment parameters from an input file
fin = open("experiments.in", "r")
for line in fin.readlines():
     tokens = line.split(" ")
     x = int(tokens[0])
     y = int(tokens[1])
     z = int(tokens[2])
     print(x, " ", y, " ", z)

fin.close()

# #Getting the mobilenet model from the keras collections
# mobilenet = tf.keras.applications.mobilenet.MobileNet()

# #Switching last layers to a dense layer with the activation function for the asl dataset
# switchLastLayers = 23
# pretrainedLayers = mobilenet.layers[-switchLastLayers].output
# reshapeOutput = tf.keras.layers.Reshape((512, ))(tf.keras.layers.GlobalAveragePooling2D()(pretrainedLayers))
# output = Dense(units=29, activation="softmax")(reshapeOutput)

# #Getting the final model from the pretrained mobilenet model
# mobile = tf.keras.Model(inputs = mobilenet.input, outputs = output)

# #Fixing the first couple layers for training
# trainableLayerNumber = 1
# for layer in mobile.layers[:-trainableLayerNumber]:
#     layer.trainable = False


# #Compiling
# mobile.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=['accuracy'])

# #training model
# mobile.fit(train_dataset, batch_size=batchSize, epochs=epochs)

# # evaluating model
# mobile.evaluate(test_dataset)

# #Calculating the elapsed time for predicitons
# #Making an initial prediction for the tf.function to initialize
# mobile.predict(x=train_dataset[0][0])

# start = time.time()
# mobile.predict(x=train_dataset[1][0])
# mobile.predict(x=train_dataset[2][0])
# mobile.predict(x=train_dataset[3][0])
# mobile.predict(x=train_dataset[4][0])
# end = time.time()
# print("Average time: ", (end-start)/5, " s")
