# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:43:10 2019

@author: isstjh
"""


import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical



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
    circ=plt.Circle((10,10),radius=10.5,color='red',fill=False)

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
print(trDat)
print(len(trDat))
print(trDat[0])
print(len(trDat[0]))

#trLbl       = np.load('kmnist-train-labels.npz')['arr_0']
trLbl=label_train
print(trLbl)
print(len(trLbl))
#tsDat       = np.load('kmnist-test-imgs.npz')['arr_0']
#tsLbl       = np.load('kmnist-test-labels.npz')['arr_0']
tsDat=twoDarray_test
tsLbl=label_test
print(len(tsLbl))

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
seed        = 22 #29
np.random.seed(seed)


modelname   = 'wks5_5'
                            # define the deep learning model
def createModel():
    model = Sequential()       
    #model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), activation='relu'))       
    model.add(Conv2D(20, (5, 5), input_shape=(21, 21, 1), activation='relu'))       
    model.add(MaxPooling2D(pool_size=(2, 2)))       
    model.add(Conv2D(40, (5, 5), activation='relu'))       
    model.add(MaxPooling2D(pool_size=(2, 2)))       
    model.add(Dropout(0.2))       
    model.add(Flatten())       
    model.add(Dense(128, activation='relu'))       
    model.add(Dense(num_classes, activation='softmax'))            
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
      
    
    return model




                            # Setup the models
model       = createModel() # This is meant for training
modelGo     = createModel() # This is used for final testing

model.summary()

# .............................................................................


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



# .............................................................................


                            # Fit the model
                            # This is where the training starts
model.fit(trDat, 
          trLbl, 
          validation_data=(tsDat, tsLbl), 
          epochs=60, 
          batch_size=32,
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
    clsname   = ['Normal Wafer','Wafer Saw Problem','Wafer Grinding Problem','Wafer Saw+Grinding Problem','Dielectric Issue','Saw+Dielectric Issue','Grinding+Dielectric Issue','Saw+Grinding+Dielectric Issues','Unknown','Scratch']
    
    if np.size(labels.shape) == 2:
        lbl  = np.argmax(labels[item])
    else:
        lbl  = labels[item]
        
    txt     = 'Class ' + str(lbl) + ': ' + clsname[lbl]     

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

plotword(35)

