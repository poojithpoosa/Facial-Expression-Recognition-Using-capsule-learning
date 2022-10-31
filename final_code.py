import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Conv2D,Input,Dense,Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from CapsNetKeras.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import cv2
import os

K.set_image_data_format('channels_last')

def CapsuleNetworkClass(input_shapes, num_class,num_routes,bs):
    padding_type='valid'
    input_layer = Input(shape=input_shapes, batch_size=bs)
    convolutional1 = Conv2D(filters=256, kernel_size=9, activation='relu', name='convolutional_layer1')(input_layer)
    capsule_primary = PrimaryCap(convolutional1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding=padding_type)
    expression_caps = CapsuleLayer(num_capsule=num_class, dim_capsule=16, routings=num_routes, name='expression_caps')(capsule_primary)

    output_caps = Length(name='capsulenetwork')(expression_caps)

    input_layer2 =Input(shape=(num_class,))
    masked_by_y = Mask()([expression_caps, input_layer2])
    masked = Mask()(expression_caps) 

    decoder_network = models.Sequential(name='decoder')
    decoder_network.add(Dense(512, activation='relu', input_dim=16 * num_class))
    decoder_network.add(Dense(256, activation='relu'))
    decoder_network.add(Dense(np.prod(input_shapes), activation='sigmoid'))
    decoder_network.add(Reshape(target_shape=input_shapes, name='out_recontructor'))
    
    train_model = models.Model([input_layer, input_layer2], [output_caps, decoder_network(masked_by_y)])
    eval_model = models.Model(input_layer, [output_caps, decoder_network(masked)])

    noise_input = Input(shape=(num_class, 16))
    noised_expressioncaps = layers.Add()([expression_caps, noise_input])
    masked_noised_y = Mask()([noised_expressioncaps, input_layer2])
    manipulate_model = models.Model([input_layer, input_layer2, noise_input], decoder_network(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train_model(capsule_model,image_data):

    (x_train, y_train), (x_test, y_test) = image_data
    capsule_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics=['acc'])

    values=capsule_model.fit([x_train, y_train], [y_train, x_train], batch_size=4, epochs=60)
    capsule_model.save_weights('/trained_model.h5')
    print('saving trained model')
    return capsule_model,values



def load_images(split=.10,path='dataset/'):
    path=path
    labels=os.listdir(path)
    image_data=[]
    image_labels=[]
    for lab in labels:
        image_path=os.listdir(path+lab+'/')
        for files in image_path:
            x=cv2.imread(path+lab+'/'+files,0)
            x=cv2.resize(x,(28,28))
            x = cv2.medianBlur(x,5)
            th3 = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            image_data.append(x)
            image_labels.append(lab)

    image_data=np.asarray(image_data).astype('float32')/255.
    image_data=np.reshape(image_data,(image_data.shape[0],image_data.shape[1],image_data.shape[2],1)).astype('float32')
    print(image_data.shape[0])

    for i in range(len(image_labels)):
        if image_labels[i] ==labels[0]:
            image_labels[i]=0
        if image_labels[i] ==labels[1]:
            image_labels[i]=1
        if image_labels[i] ==labels[2]:
            image_labels[i]=2
        if image_labels[i] ==labels[3]:
            image_labels[i]=3
        if image_labels[i] ==labels[4]:
            image_labels[i]=4
        if image_labels[i] ==labels[5]:
            image_labels[i]=5
    image_labels = to_categorical(image_labels).astype('float32')
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(image_data,image_labels,test_size=split,shuffle=True)
    return (x_train,y_train),(x_test,y_test)

(x_train, y_train), (x_test, y_test) = load_images()
model, eval_model, manipulate_model = CapsuleNetworkClass(input_shapes=x_train.shape[1:],
                                                  num_class=len(np.unique(np.argmax(y_train, 1))),
                                                  num_routes=3,
                                                  bs=4)
model.summary()
capsule_model,values=train_model(capsule_model=model, image_data=((x_train, y_train), (x_test, y_test)))

def testing(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test,batch_size=4)
    print('Test accuracy:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

(x_test,y_test),_=load_images(0.01,'dataset_test/')

testing(eval_model,(x_test,y_test))

loss=values.history['loss']
accuracy=values.history['capsulenetwork_acc']

plt.title('loss plot on training')
plt.plot(loss,label='loss curve')
plt.ylabel('MSE loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.title('accuracy plot on training')
plt.plot(accuracy,label='accuracy curve')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()

def predict(img):
    image_data=[]
    l=os.listdir(img)
    
    for files in l:
        print(img+'/'+files)
        x=cv2.imread(img+'/'+files,0)
        print(x)
        x=cv2.resize(x,(28,28))
        #x = cv2.medianBlur(x,5)
        #th3 = cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        image_data.append(x)
    x=cv2.imread(img,0)
    x=cv2.resize(x,(28,28))
    x=np.reshape(x,(1,28,28,1)).astype('float32')
    eval_model.predict(x,batch_size=4)


predict("dataset_test/anger")

