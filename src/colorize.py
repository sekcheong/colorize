import os
import cv2
import keras
import numpy as np
import tensorflow as tf
import time
import datetime
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras import losses
from keras import callbacks
from keras import backend as K


config = {

    'imageWidth'                  : 224,
    'imageHeight'                 : 224,
    'colorDownSamplingRate'       : 0.15,

    'epochsToRun'                 : 1,
    'learnRate'                   : 0.05,
    'decay'                       : 4e-5,
    'momentum'                    : 0.9,
    'batchSize'                   : 50,
    'dropoutRate'                 : 0.5,
    
    'samplePrecent'               : 1500/7000,
    'checkpointPeriod'            : 5,
    'randomSeed'                  : 838*838,
}


def computeColorChannelSize(imageWidth=224, imageHeight=224, downSamplingRate=config['colorDownSamplingRate']):
    return 2*(int(imageWidth*downSamplingRate)*int(imageHeight*downSamplingRate))


def makeColorizeModel(imageWidth=224, imageHeight=224, downSamplingRate=config['colorDownSamplingRate']):
    
    colorChannelSize = computeColorChannelSize(imageWidth, imageHeight, downSamplingRate)

    print("color channel size:", colorChannelSize)    

    model = Sequential()   
    
    # The input layer
    model.add(InputLayer(input_shape = [imageHeight, imageWidth, 3]))

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Final FC layers
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(1000, activation='softmax', name='output'))
    #model.add(Dense(224*224*2, activation='tanh', name='output'))
    model.summary()

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


def imageCompressColorSpace(Y, U, V, ratio=config['colorDownSamplingRate']):
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


def makeOneExample(img):
    (Y, U, V) = imageRGB2YUV(img)
    (Y, U, V) = imageCompressColorSpace(Y, U, V)

    #use the intensity as input image
    x = imageFromComponent(Y)
    
    #convert the U, V components to 1D arrays
    U = U.ravel()
    V = V.ravel()
    y = np.append(U, V)
    y = normalizeImage(y)    
    return x, y


def unpreprocessImage(x):
    # un Zero-center by mean pixel
    x = np.copy(x)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    return x.astype(dtype='uint8', copy=False)


def makeExamples(images):
    (exampleX, exampleY) = ([], [])
    for img in images:
        (x, y) = makeOneExample(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(K.floatx(), copy=False)
        x = preprocess_input(x)        
        exampleX.append(x)        
        exampleY.append(y)

    #convert the example list to column vector format for training using Keras
    exampleX = np.vstack(tuple(exampleX))
    #print("ex:",exampleX.shape)
    exampleY = np.vstack(tuple(exampleY))

    return (exampleX, exampleY)


## Normalize the pixel value between [-1, 1]
#  @param img: The image in numpy array
#  @return: Returns normalized image
def normalizeImage(img):
    img = (img/255.0)*2-1
    return img.astype(dtype='float32', copy=False)


def denormalizeImage(img):
    img = ((img+1)/2.0)*255
    return img.astype(dtype='uint8', copy=False)


## Reconstruct a image Y from intensity vector and chromiance 
## vector UY concated togeter
#  @param x: The intensity vector of the image
#  @param y: A [2, W, H] vector contains the chrominance of the image
#  @return: A fully reconstructed image 
def reconstructImage(x, y):
    y = denormalizeImage(y)
    y = np.hsplit(y, 2)
    U = y[0].reshape(33, 33)
    V = y[1].reshape(33, 33)
    Y = x[:,:,0]
    (Y, U, V) = imageDecompressColorSpace(Y, U, V)
    img = imageYUV2RGB(Y, U, V)
    return img


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
   
    images = loadOneImageSet(trainPath, ratio)
    (trainX, trainY) = makeExamples(images)

    images = loadOneImageSet(tunePath, ratio)
    (tuneX, tuneY) = makeExamples(images)

    images = loadOneImageSet(testPath, ratio)
    (testX, testY) = makeExamples(images)
    
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


## The mean square error loss function 
#  @param y_ture: The ground truth of the value
#  @param y_pred: The predicted value
#  @return: Returns a sentence with your variables in it
def mseLoss(y_true, y_pred):    
    return K.mean(K.square(y_pred - y_true), axis=-1)


def maeLoss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def euclideanLoss(y_true, y_pred):
    u0, v0 = tf.split(y_true, 2)
    u1, v1 = tf.split(y_pred, 2)
    e = tf.sqrt(tf.norm(u0-v0)+tf.norm(u1-v1))
    return e


def kullback_leibler_divergence(y_true, y_pred):
    hist_true = tf.histogram_fixed_width(y_true, [-1.0, 1.0], nbins=256, dtype=K.floatx())
    hist_pred = tf.histogram_fixed_width(y_pred, [-1.0, 1.0], nbins=256, dtype=K.floatx())

    hist_true = hist_true / (33*33.0*2)
    hist_pred = hist_pred / (33*33.0*2)

    hist_true = K.clip(hist_true, K.epsilon(), 1)
    hist_pred = K.clip(hist_pred, K.epsilon(), 1)


    #b = hist_pred.eval()

    # W1 = tf.ones((2,2))
    # W2 = tf.Variable(tf.zeros((2,2)), name="weights")
    # with K.get_session()  as sess:
    #     print(sess.run(hist_pred))
    #     sess.run(tf.initialize_all_variables())
    #     print(sess.run(W2))

    #with K.get_session()():
    # with K.get_session() as sess:
    #     b = hist_true.eval()
    #     open('test.txt', 'w').write(str(b))

    #b = hist_true.eval(session=K.get_session())  # Runs to get the output of 'a' and converts it to a numpy array
   
    # y_t = K.clip(hist_true, K.epsilon(), 1)
    # y_p = K.clip(hist_pred, K.epsilon(), 1)
    # #print(hist_true.shape)
    # sess = K.get_session()
    # z = sess.run(y_t)
    # open('test.txt', 'w').write(str(y_t))
    #y_true = K.clip(y_true, K.epsilon(), 1)
    #y_pred = K.clip(y_pred, K.epsilon(), 1)
    #return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    return K.mean(K.square(hist_pred - hist_true), axis=-1)
    

def histoLoss(y_true, y_pred):
    hist_true = tf.histogram_fixed_width(y_true, [-1.0, 1.0], nbins=256, dtype=K.floatx())
    hist_pred = tf.histogram_fixed_width(y_pred, [-1.0, 1.0], nbins=256, dtype=K.floatx())

    hist_true = hist_true / (33*33.0*2)
    hist_pred = hist_pred / (33*33.0*2)
        
    hist_true = K.clip(hist_true, K.epsilon(), 1)
    hist_pred = K.clip(hist_pred, K.epsilon(), 1)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(hist_pred, hist_true))), name="rmse")
    return rmse


def trainColorizeModel(model, trainX, trainY, tuneX, tuneY):
    model.pop()
    model.pop()
    model.pop()

    model.add(
        Dropout(config['dropoutRate'])
    )

    model.add(
        Dense(
            units = 1536,    
            kernel_initializer = keras.initializers.lecun_uniform(seed=None), 
            name = 'fc1'
        )
    )

    model.add(
        LeakyReLU(alpha=0.3)
    )

    model.add(
        Dropout(config['dropoutRate'])
    )

    model.add(
        Dense(
            units = 1536,            
            kernel_initializer = keras.initializers.lecun_uniform(seed=None), 
            name = 'fc2'
        )
    )

    model.add(
        LeakyReLU(alpha=0.3)
    )

    model.add(
        Dropout(config['dropoutRate'])
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dense(
            units = 2178,             
            kernel_initializer = keras.initializers.lecun_uniform(seed=None), 
            name = 'colorize'
        )
    )

    model.add(
        Activation('tanh')
    )

    #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)    
    #opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0008, nesterov=False)
    opt = keras.optimizers.SGD(
        lr       = config['learnRate'],
        decay    = config['decay'],
        momentum = config['momentum'],
        nesterov = True
    )    

    
    model.compile(
        optimizer = opt,
        loss      = histoLoss, #kullback_leibler_divergence,
        metrics   = ['accuracy']
    )    

    model.summary()

    history = model.fit(trainX, trainY,
                validation_data = (tuneX, tuneY),
                epochs     = config['epochsToRun'],
                batch_size = config['batchSize'],
                shuffle    = True,
                verbose    = 1,
                callbacks  = [

                    callbacks.ModelCheckpoint(                        
                        '../models/checkpoints/colorize_weights.{epoch:02d}-{val_loss:.2f}.h5',
                        monitor           = 'val_loss',
                        verbose           = 0,
                        save_best_only    = False,
                        save_weights_only = True,
                        mode              = 'auto',
                        period            = config['checkpointPeriod']
                    ),

                    # callbacks.EarlyStopping(
                    #      monitor           = 'val_loss',                         
                    #      patience          = 2, 
                    #      verbose           = 0,
                    #      mode              = 'auto'
                    # )

                ]
              )

    # save the model 
    json = model.to_json()
    open('../models/colorize_model.json', 'w').write(json)
    
    # save the weights
    model.save_weights('../models/colorize_weights.h5', overwrite=True)

    open('history.json', 'w').write(str(history.history))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('model_accuracy.png', dpi=600)
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')    
    plt.show()

    # get the time stamp
    #timestamp = datetime.datetime.now().strftime('%Y-%m-%d.%H%M%S')
    #model.save_weights('../models/colorize_weights.'+timestamp+'.h5')


def predictColorizeModel(model, testX, testY):

    json = open('../models/colorize_model.json').read()
    model = model_from_json(json)  
    model.summary()
    print("Load trained weights...")
    model.load_weights('../models/colorize_weights.49-0.03.h5')

    print("Colorizing images...")
    y = model.predict(testX)

    # for i in range(y.shape[0]):        
    #     z = y[i]
    #     z = denormalizeImage(z)
    #     z = np.hsplit(z, 2)
    #     U = z[0].reshape(33, 33)
    #     V = z[1].reshape(33, 33)
    #     print('U=',U)
    #     print('V=',V)

    for i in range(y.shape[0]):        
        x = unpreprocessImage(testX[i])
        example = x
        predict = reconstructImage(x, y[i])
        groundTruth = reconstructImage(x, testY[i])
        img = imageConcat([groundTruth, example, predict])
        cv2.imwrite('../images/out/im{:d}.jpg'.format(i), img)
        
        #preview the last image
        if (i==y.shape[0]-1):
            imageShow(img)


def colorize(model, x, y):
    img = reconstructImage(x, y)
    return img




if __name__ == '__main__':

    np.random.seed(838*838)

    print("Loading training data set...")
    (trainX, trainY), (tuneX, tuneY), (testX, testY) = loadDataSet('../images', config['samplePrecent'])

    print ("Train examples:", len(trainX))
    print ("Tune examples :", len(tuneX))
    print ("Test examples :", len(testX))
    
    model = makeColorizeModel()
    model.summary()

    trainColorizeModel(model, trainX, trainY, tuneX, tuneY)
    #predictColorizeModel(model, testX, testY)



