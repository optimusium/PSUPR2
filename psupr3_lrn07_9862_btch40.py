# -*- coding: utf-8 -*-

"""

Created on Sat Sep 14 14:28:46 2019



@author: Ong Boon Ping (A0195172B), Tan Chin Gee (A0195296M), & Han Dongchou Francis (A0195414A) 

"""



import numpy as np

import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import os

import pandas as pd

from sklearn.model_selection import train_test_split





from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D

from tensorflow.keras.layers import add

from tensorflow.keras.regularizers import l2

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import cifar10

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator





                            # Setting up the font manager, so that

                            # it can show japanese characters correctly

from matplotlib import font_manager as fm

fpath       = os.path.join(os.getcwd(), "ipam.ttf")

prop        = fm.FontProperties(fname=fpath)





                            # Set up 'ggplot' style

plt.style.use('ggplot')     # if want to use the default style, set 'classic'

plt.rcParams['ytick.right']     = True

plt.rcParams['ytick.labelright']= True

plt.rcParams['ytick.left']      = False

plt.rcParams['ytick.labelleft'] = False

plt.rcParams['font.family']     = 'Arial'







                            # Create a functin do plot gray easily

def grayplt(img,title=''):

    '''

    plt.axis('off')

    if np.size(img.shape) == 3:

        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)

    else:

        plt.imshow(img,cmap='gray',vmin=0,vmax=1)

    plt.title(title, fontproperties=prop)

    '''

    

    fig,ax = plt.subplots(1)

    ax.set_aspect('equal')

    





    # Show the image

    if np.size(img.shape) == 3:

        ax.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)

    else:

        ax.imshow(img,cmap='gray',vmin=0,vmax=1)

    circ=plt.Circle((10,10),radius=10.1,color='red',fill=False,linewidth=2.5)



    ax.add_patch(circ)

    

    plt.show()





# .............................................................................

#raw_data = pd.read_excel('td5.xlsx',sheet_name = "td5")

raw_data = pd.read_excel('td6.xlsx',sheet_name = "td6")

sdarray=raw_data.drop(["saw","grind","dielectric"],axis=1)

raw_label=raw_data.saw+2*raw_data.grind+4*raw_data.dielectric



sdarray_train,sdarray_test,label_train,label_test = train_test_split(sdarray,raw_label,test_size = 0.2)



sdarray_train_np=sdarray_train.as_matrix()



twoDarray_train = sdarray_train_np.reshape((sdarray_train_np.shape[0], 21, 21))

#print(twoDarray_train)

#print(len(twoDarray_train[0][0]))

sdarray_test_np=sdarray_test.as_matrix()



twoDarray_test = sdarray_test_np.reshape((sdarray_test_np.shape[0], 21, 21))





#raise

#X_train,X_test,y_train,y_test = train_test_split(sdarray,raw_label,test_size = 0.2)





                            # Load the data

#trDat       = np.load('kmnist-train-imgs.npz')['arr_0']

trDat=twoDarray_train   

'''                         

print(trDat)

print(len(trDat))

print(trDat[0])

print(len(trDat[0]))

'''



#trLbl       = np.load('kmnist-train-labels.npz')['arr_0']

trLbl=label_train

'''

print(trLbl)

print(len(trLbl))

'''

#tsDat       = np.load('kmnist-test-imgs.npz')['arr_0']

#tsLbl       = np.load('kmnist-test-labels.npz')['arr_0']

tsDat=twoDarray_test

tsLbl=label_test

#print(len(tsLbl))



#raise

                            # Convert the data into 'float32'

                            # Rescale the values from 0~255 to 0~1

#trDat       = trDat.astype('float32')/5 #/255

#tsDat       = tsDat.astype('float32')/5 #/255

trDat       = trDat.astype('float32')/500 #/255

tsDat       = tsDat.astype('float32')/500 #/255





                            # Retrieve the row size of each image

                            # Retrieve the column size of each image

imgrows     = trDat.shape[1]

imgclms     = trDat.shape[2]





                            # reshape the data to be [samples][width][height][channel]

                            # This is required by Keras framework

trDat       = trDat.reshape(trDat.shape[0],

                            imgrows,

                            imgclms,

                            1)

tsDat       = tsDat.reshape(tsDat.shape[0],

                            imgrows,

                            imgclms,

                            1)





                            # Perform one hot encoding on the labels

                            # Retrieve the number of classes in this problem

trLbl       = to_categorical(trLbl)

tsLbl       = to_categorical(tsLbl)

num_classes = tsLbl.shape[1]













# .............................................................................



                            # fix random seed for reproducibility

seed        = 29

np.random.seed(seed)



'''

modelname   = 'wks5_5'

                            # define the deep learning model

def createModel():

    model = Sequential()       

    #model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation='relu'))       

    model.add(Conv2D(20, (4, 4), input_shape=(21, 21, 1), activation='relu'))       

    #model.add(MaxPooling2D(pool_size=(2, 2)))       

    model.add(MaxPooling2D(pool_size=(2, 2)))       

    model.add(Conv2D(40, (4, 4), activation='relu'))       

    #model.add(MaxPooling2D(pool_size=(2, 2)))       

    model.add(MaxPooling2D(pool_size=(2, 2)))       

  

    model.add(Dropout(0.2))       

    model.add(Flatten())       

    model.add(Dense(128, activation='relu'))       

    model.add(Dense(num_classes, activation='softmax'))            

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

      

    

    return model

'''



#optmz       = optimizers.Adam(lr=0.001)
optmz       = optimizers.Adam(lr=0.0008)

modelname   = 'cifar10ResV1Cfg5'

                            # define the deep learning model





def resLyr(inputs,

           numFilters=16,

           kernelSz=3,

           strides=1,

           activation='relu',

           batchNorm=True,

           convFirst=True,

           lyrName=None):



    convLyr     = Conv2D(numFilters,  kernel_size=kernelSz, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name=lyrName+'_conv' if lyrName else None) 

    x           = inputs 

    if convFirst:           

        x       = convLyr(x)           

        if batchNorm:               x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)           

        if activation is not None:               x   = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)       

    else:           

        if batchNorm:               x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)           

        if activation is not None:               x   = Activation(activation,                              name=lyrName+'_'+activation if lyrName else None)(x)           

        x       = convLyr(x)     

    return x



def resLyrMax(inputs,

           numFilters=16,

           kernelSz=3,

           strides=1,

           activation='relu',

           batchNorm=True,

           convFirst=True,

           lyrName=None):



    convLyr     = Conv2D(numFilters,  kernel_size=kernelSz, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name=lyrName+'_conv' if lyrName else None) 

    maxLyr=MaxPooling2D(pool_size=(2,2), padding='same')

    x           = inputs 

    if convFirst:           

        x       = convLyr(x)

        x       = maxLyr(x)           

        if batchNorm:               x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)           

        if activation is not None:  x   = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)       

    else:           

        if batchNorm:               x   = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)           

        if activation is not None:  x   = Activation(activation,                              name=lyrName+'_'+activation if lyrName else None)(x)           

        x       = maxLyr(x)     

    return x



def resBlkV1(inputs,

             numFilters=16,

             numBlocks=3,

             downsampleOnFirst=True,

             names=None):



    x           = inputs       

    for run in range(0,numBlocks):           

        strides = 1           

        blkStr  = str(run+1)            

        if downsampleOnFirst and run == 0:               

            strides     = 2                   

        y       = resLyr(inputs=x, numFilters=numFilters, strides=strides, lyrName=names+"y1_%s"%run)           

        y       = resLyr(inputs=y, numFilters=numFilters, activation=None, lyrName=names+"y2_%s"%run)              

        if downsampleOnFirst and run == 0:               

            x   = resLyr(inputs=x, numFilters=numFilters, kernelSz=1, strides=strides, activation=None, batchNorm=False, lyrName=names+"x0_%s"%run)           

        x       = add([x,y],                         name=names+"x1_%s"%run)           

        x       = Activation('relu',                                name=names+"x2_%s"%run)(x)             

    return x

    

def resBlkV1Max(inputs,

             numFilters=16,

             numBlocks=3,

             downsampleOnFirst=True,

             names=None):



    x           = inputs       

    for run in range(0,numBlocks):           

        strides = 1           

        blkStr  = str(run+1)            

        if downsampleOnFirst and run == 0:               

            #    strides     = 2                   

            y       = resLyrMax(inputs=x, numFilters=numFilters, strides=strides, lyrName=names+"y1_%s"%run)

        else:

            y       = resLyr(inputs=x, numFilters=numFilters, strides=strides, lyrName=names+"y1_%s"%run)           

        y       = resLyr(inputs=y, numFilters=numFilters, activation=None, lyrName=names+"y2_%s"%run)              

        if downsampleOnFirst and run == 0:               

            x   = resLyrMax(inputs=x, numFilters=numFilters, kernelSz=1, strides=strides, activation=None, batchNorm=False, lyrName=names+"x0_%s"%run)           

        x       = add([x,y],                         name=names+"x1_%s"%run)           

        x       = Activation('relu',                                name=names+"x2_%s"%run)(x)             

    return x

    





def createResNetV1(inputShape=(21,21,1),

                   numClasses=8):

    

    inputs      = Input(shape=inputShape)       

    v           = resLyr(inputs,                            lyrName='Inpt') 

          

    v           = resBlkV1(inputs=v, numFilters=16,  numBlocks=3, downsampleOnFirst=False, names='Stg1')       

    v           = resBlkV1Max(inputs=v, numFilters=32,  numBlocks=3, downsampleOnFirst=True,  names='Stg2')       

    v           = resBlkV1(inputs=v, numFilters=64,  numBlocks=3, downsampleOnFirst=True,  names='Stg3')       

    v           = resBlkV1(inputs=v, numFilters=128,  numBlocks=4, downsampleOnFirst=True,  names='Stg4')       

    v           = AveragePooling2D(pool_size=2,  name='AvgPool')(v)       

    #v           = AveragePooling2D(pool_size=4,  name='AvgPool')(v)       

    v           = Flatten()(v)       

    

    outputs     = Dense(numClasses, activation='softmax', kernel_initializer='he_normal')(v)   

        

    model       = Model(inputs=inputs,outputs=outputs)       

    model.compile(loss='categorical_crossentropy', optimizer=optmz, metrics=['accuracy'])   

    

    

    return model



model       = createResNetV1()  # This is meant for training

modelGo     = createResNetV1()  # This is used for final testing



                            # Setup the models

#model       = createModel() # This is meant for training

#modelGo     = createModel() # This is used for final testing



model.summary()



#raise

# .............................................................................



def lrSchedule(epoch):

    lr  = 0.7e-3

    

    if epoch > 160:

        lr  *= 1e-4

        

    elif epoch > 140:

        lr  *= 1e-3

        

    elif epoch > 120:

        lr  *= 1e-2

        

    elif epoch > 80:

        lr  *= 1e-1

        

    print('Learning rate: ', lr)

    

    return lr



LRScheduler     = LearningRateScheduler(lrSchedule)



                            # Create checkpoint for the training

                            # This checkpoint performs model saving when

                            # an epoch gives highest testing accuracy

filepath        = modelname + ".hdf5"

checkpoint      = ModelCheckpoint(filepath, 

                                  monitor='val_acc', 

                                  verbose=0, 

                                  save_best_only=True, 

                                  mode='max')



                            # Log the epoch detail into csv

csv_logger      = CSVLogger(modelname +'.csv')

callbacks_list  = [checkpoint,csv_logger,LRScheduler]




'''


                            # Create checkpoint for the training

                            # This checkpoint performs model saving when

                            # an epoch gives highest testing accuracy

filepath        = modelname + ".hdf5"

checkpoint      = ModelCheckpoint(filepath, 

                                  monitor='val_acc', 

                                  verbose=0, 

                                  save_best_only=True, 

                                  mode='max')



                            # Log the epoch detail into csv

csv_logger      = CSVLogger(modelname +'.csv')

callbacks_list  = [checkpoint,csv_logger]

'''





# .............................................................................





                            # Fit the model

                            # This is where the training starts

'''

model.fit(trDat, 

          trLbl, 

          validation_data=(tsDat, tsLbl), 

          epochs=1 #60, 

          batch_size=32,

          callbacks=callbacks_list)

'''

datagen = ImageDataGenerator(width_shift_range=0.1,

                             height_shift_range=0.1,

                             rotation_range=20,

                             horizontal_flip=True,

                             vertical_flip=False)



model.fit_generator(datagen.flow(trDat, trLbl, batch_size=40),

                    validation_data=(tsDat, tsLbl),

                    epochs=200, 

                    verbose=1,

                    steps_per_epoch=len(trDat)/40,

                    callbacks=callbacks_list)









# ......................................................................





                            # Now the training is complete, we get

                            # another object to load the weights

                            # compile it, so that we can do 

                            # final evaluation on it

modelGo.load_weights(filepath)

modelGo.compile(loss='categorical_crossentropy', 

                optimizer='adam', 

                metrics=['accuracy'])



 







# .......................................................................





                            # Make classification on the test dataset

predicts    = modelGo.predict(tsDat)





                            # Prepare the classification output

                            # for the classification report

predout     = np.argmax(predicts,axis=1)

testout     = np.argmax(tsLbl,axis=1)

#labelname   = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']

labelname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']

                                            # the labels for the classfication report





testScores  = metrics.accuracy_score(testout,predout)

confusion   = metrics.confusion_matrix(testout,predout)





print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))

print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))

print(confusion)





# ...................................................................







def plotword(item,data=trDat,labels=trLbl):

    #clsname  = ['お O','き Ki','す Su','つ Tsu','な Na','は Ha','ま Ma','や Ya','れ Re','を Wo']

    #clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']

    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues']

    

    if np.size(labels.shape) == 2:

        lbl  = np.argmax(labels[item])

    else:

        lbl  = labels[item]

        

    txt     = 'Class ' + str(lbl) + ': ' + clsname[lbl]     

    print(txt)

    grayplt(data[item],title=txt)

    

    

    

# ..................................................................

    

import pandas as pd



records     = pd.read_csv(modelname +'.csv')

plt.figure()

plt.subplot(211)

plt.plot(records['val_loss'])

plt.yticks([0.00,0.10,0.20,0.30])

plt.title('Loss value',fontsize=12)



ax          = plt.gca()

ax.set_xticklabels([])







plt.subplot(212)

plt.plot(records['val_acc'])

plt.yticks([0.93,0.95,0.97,0.99])

plt.title('Accuracy',fontsize=12)

plt.show()



exampled=[]

for i in range(len(trLbl)):

    #print(trLbl[i])

    for j in range(len(trLbl[i])):

        if trLbl[i][j]==1:

            break

    if j not in exampled:

        plotword(i)

        exampled.append(j)

    if len(exampled)==8: 

        break

        

'''

plotword(35)

plotword(235)

plotword(835)

plotword(635)

plotword(435)

'''