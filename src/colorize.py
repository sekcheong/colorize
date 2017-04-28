import os
import numpy as np
import warnings
import time
import cv2
import keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import losses
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


config = {

    'imageWidth'  : 224,
    'imageHeight' : 224,
    'downSample'  : 0.15,
    'epochsToRun' : 10,
    'batchSize'   : 5

}


def euclideanLoss(y_true, y_pred):
    d = tf.constant(3.0, tf.float32)

    # U1 = tf.constant(3.0, tf.float32)
    # V1 = tf.constant(3.0, tf.float32)
    
    # U2 = tf.constant(3.0, tf.float32)
    # V2 = tf.constant(3.0, tf.float32)

     # Euclidean distance between x1,x2
    l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)), reduction_indices=1))    
    return l2


def makeColorizeModel(imageWidth=224, imageHeight=224, downSamplingRate=config['downSample']):
    
    colorChannelSize = 2*(int(imageWidth*downSamplingRate)*int(imageHeight*downSamplingRate))

    #print("color size:", colorChannelSize)

    inputShape = _obtain_input_shape((imageWidth, imageHeight, 3), default_size=imageWidth,
                                                                   min_size=48,
                                                                   data_format=K.image_data_format(),
                                                                   include_top=False)    
    
    imgInput = Input(shape=inputShape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(imgInput)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Final FC layers
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out = Dense(colorChannelSize, activation='sigmoid', name='colorize')(x)
    
    model = Model(imgInput, out)

    return model


#image processing related functions

def imageLoad(path):
    return cv2.imread(path)


def imageRGB2YUV(img):
    out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y = out[:,:,0]
    U = out[:,:,1]
    V = out[:,:,2]
    return (Y, U, V)


def imageYUV2RGB(Y, U, V):
    if (U.shape != Y.shape):
        U = cv2.resize(U, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
    if (V.shape != Y.shape):
        V = cv2.resize(V, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
    yuv = (np.dstack([Y,U,V])).astype(np.uint8)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return img


def imageCompressColorSpace(Y, U, V, ratio=config['downSample']):
    U = cv2.resize(U, (int(U.shape[1]*ratio), int(U.shape[0]*ratio) ) , interpolation = cv2.INTER_CUBIC)
    V = cv2.resize(V, (int(V.shape[1]*ratio), int(V.shape[0]*ratio) ) , interpolation = cv2.INTER_CUBIC)
    return (Y, U, V)


def imageDecompressColorSpace(Y, U, V):
    U = cv2.resize(U, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
    V = cv2.resize(V, (Y.shape[1], Y.shape[0]), interpolation = cv2.INTER_CUBIC)
    return (Y, U, V)


def imageFromComponent(data):
    out = np.zeros((data.shape[0], data.shape[1], 3), dtype='uint8')
    out[:, :, 0] = data[:, :]
    out[:, :, 1] = data[:, :]
    out[:, :, 2] = data[:, :]
    return out;


def imageConcat(images):
    n = len(images)
    w = images[0].shape[1]
    h = images[0].shape[0]
    totalW = w * n
    newImage = np.zeros(shape=(h, totalW, 3), dtype=np.uint8)
    pos = 0
    for img in images:
        newImage[:h, pos:pos+w]=img
        pos = pos + w
    return newImage


def makeTrainExample(img):
    (Y, U, V) = imageRGB2YUV(img)
    (Y, U, V) = imageCompressColorSpace(Y, U, V)

    #use the intensity as input image
    x = imageFromComponent(Y)

    #imageShow(Y)
    #imageShow(U)
    #imageShow(V)
    
    #convert the U, V components to 1D array
    U = U.ravel()
    V = V.ravel()
    y = np.append(U,V)    
    return x, y


def isImageFile(name):
    fn = name.lower()
    return name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg")


def loadImages(path):
    images = []
    for fname in os.listdir(path):
        if isImageFile(fname):
            imgPath = os.path.join(path, fname)
            img = imageLoad(imgPath)
            images.append(img)
    return images


def loadOneImageSet(path, ratio=1.0):
    fileList = [f for f in os.listdir(path) if isImageFile(f)]
    fileList = fileList[:int(len(fileList)*ratio)]
    
    images = []
    for fname in fileList:
        if isImageFile(fname):            
            imgPath = os.path.join(path, fname)            
            img = imageLoad(imgPath)
            images.append(img)
    return images


def loadDataSet(basePath, ratio=1.0):
    trainPath = os.path.join(basePath, 'train')
    tunePath  = os.path.join(basePath, 'tune')
    testPath  = os.path.join(basePath, 'test')

    (trainX, trainY, tuneX, tuneY, testX, testY) = ([], [], [], [], [], [])
   
    for img in loadOneImageSet(trainPath, 0.1):
        (x, y) = makeTrainExample(img)
        trainX.append(x)
        trainY.append(y)

    for img in loadOneImageSet(tunePath, 0.1):
        (x, y) = makeTrainExample(img)
        tuneX.append(x)
        tuneY.append(y)

    for img in loadOneImageSet(testPath, 0.1):
        (x, y) = makeTrainExample(img)
        testX.append(x)
        testY.append(y)
    
    return (trainX, trainY), (tuneX, tuneY), (testX, testY) 




def imageShow(img):
    cv2.imshow('image', img)
    k = cv2.waitKey(0) & 0xFF

    # wait for ESC key to exit
    if k == 27:         
        cv2.destroyAllWindows()
        # wait for 's' key to save and exit
    elif k == ord('s'): 
        cv2.imwrite('out.png',img)
        cv2.destroyAllWindows()




if __name__ == '__main__':

    print("Creating model...", end='', flush=True)
    start = time.time()
    #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
    end = time.time()
    print("done. ", "elapsedTime=", end-start)

    model = makeColorizeModel()
    model.compile(optimizer = opt,
                  loss='mean_squared_error',
                  metrics = ['accuracy'])

    start = time.time()
    print("Loading data set...", end='', flush=True)
    (trainX, trainY), (tuneX, tuneY), (testX, testY) = loadDataSet('../images')
    end = time.time()
    print("done. ", "elapsedTime=", end-start)

    print ("Train examples:", len(trainX))
    print ("Tune examples :", len(tuneX))
    print ("Test examples :", len(testX))

    
    print("Training  started...", end='', flush=True)
    for i in range(config['epochsToRun']):
        model.fit(trainX, trainY,
              epochs     = 1,
              batch_size = config['batchSize'],
              shuffle    = True)

    print("done")


    #keras.utils.plot_model(model, to_file='model.png')


    # img = imageLoad('../images/processed/im5.jpg')
    # x, y = makeTrainExample(img)

    # imageShow(x)
    # a, b = loadData()

    # print(a, ",", b)


















#END




    # print("Loading weights...")
    # model = VGG16(include_top=True, weights='imagenet')
    # print("done!")

    # for l in model.layers:
    #     for v in l.get_weights():
    #         print (v.shape)

    # plot_model(model, to_file='model.png')
    # img_path = 'cat.jpg'
    
    # img = image.load_img(img_path, target_size=(224, 224))
    
    # x = image.img_to_array(img)

    # x = np.expand_dims(x, axis=0)
    # print('input1:', x.shape)

    # x = preprocess_input(x)
    # print('input2:', x)
    # print('Input image shape:', x.shape)
    # start = time.time()
    # preds = model.predict(x)
    # print('Predicted:', np.argmax(preds))
    # end = time.time()
    # print(end - start)
