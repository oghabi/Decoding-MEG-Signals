import numpy as np
import scipy.io
from boto.s3.connection import S3Connection
import os
import random
import csv

#Check if data directory doesn't exist (To fetch the files stored on S3 (Useful for when running on EC2 for the first time))
if not os.path.isdir("./data"):
	os.makedirs('data')

	#Also install keras and tensor flow
	os.system('pip install tensorflow')
	os.system('pip install keras')

	#Access keys to use AWS SDK to access S3 Storage Bucket
	aws_access_key_id = 'AKIAI5HTONKOA2GK2OVA'
	aws_secret_access_key = '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'
	conn = S3Connection(aws_access_key_id, aws_secret_access_key)
	mybucket = conn.get_bucket('oghabi-deeplearningproject')

	print ('Downloading Files from S3')
	for object in mybucket.get_all_keys():
		subject_file = mybucket.get_key(object.key)
		print ("Downloading: " + object.key)
		#object.key is "data/train_subjectXX.mat"
		subject_file.get_contents_to_filename(object.key)
	print ("Download Completed")

else:
	print ("Data Directory Already Exists")


#To make sure we have keras and tensorflow installed
import keras
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional, Lambda, Wrapper, ConvLSTM2D, Input
from keras.layers.core import Permute, Reshape
from keras.optimizers import SGD, rmsprop
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import initializers
from keras.engine import InputSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1:])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


train_data = None
x_train = None
y_train = None

test_data = None
x_test = None
y_test = None

KaggleTesting = 1
y_id = None


Start_TrainSubject = 1
End_TrainSubject = 11
Start_TestSubject = 12
End_TestSubject = 16

print ("Kaggle Testing Is: ", KaggleTesting)

#For real testing, use all training data
if KaggleTesting:
	Start_TrainSubject = 1
	End_TrainSubject = 16
	Start_TestSubject = 17
	End_TestSubject = 23


########################################################
#Concatenating Training subjects 1 to 11
# for index,i in enumerate(range(Start_TrainSubject, End_TrainSubject+1)):
# 	#Initialize the first loading
# 	if index == 0:
# 		train_data = scipy.io.loadmat('./preproc/train_subject' + str(i).zfill(2) +'.mat')
# 		x_train = np.rollaxis(train_data['X'], 2)
# 		y_train = train_data['y']

# 	else:
# 		train_data = scipy.io.loadmat('./preproc/train_subject' + str(i).zfill(2) +'.mat')
# 		x_tr= np.rollaxis(train_data['X'], 2)
# 		y_tr = train_data['y']
# 		x_train = np.concatenate((x_train , x_tr))   #Add more training examples
# 		y_train = np.concatenate((y_train , y_tr))


# #Concatenating testing subjects 12 to 16
# for index,i in enumerate(range(Start_TestSubject, End_TestSubject+1)):
# 	#Initialize the first loading
# 	if index == 0:
# 		test_data = scipy.io.loadmat('./preproc/train_subject' + str(i).zfill(2) +'.mat')
# 		x_test = np.rollaxis(test_data['X'], 2)
# 		if not KaggleTesting:
# 			y_test = test_data['y']  #Remember real test set data doesn't have this

# 		#Merge the Id's of the testing subjects for real Kaggle Testing
# 		else: 
# 			y_id = test_data['Id']

# 	else:
# 		test_data = scipy.io.loadmat('./preproc/train_subject' + str(i).zfill(2) +'.mat')
# 		x_tst = np.rollaxis(test_data['X'], 2)
# 		x_test = np.concatenate((x_test , x_tst))   #Add more test examples

# 		if not KaggleTesting:
# 			y_tst = test_data['y']
# 			y_test = np.concatenate((y_test , y_tst))

# 		else:
# 			ids = test_data['Id']
# 			y_id = np.concatenate((y_id , ids))
########################################################


# Concatenating Training subjects 1 to 11
for index,i in enumerate(range(Start_TrainSubject, End_TrainSubject+1)):
	#Initialize the first loading
	if index == 0:
		train_data = scipy.io.loadmat('./data/train_subject' + str(i).zfill(2) +'.mat')
		x_train = train_data['X']
		y_train = train_data['y']

	else:
		train_data = scipy.io.loadmat('./data/train_subject' + str(i).zfill(2) +'.mat')
		x_tr= train_data['X']
		y_tr = train_data['y']
		x_train = np.concatenate((x_train , x_tr))   #Add more training examples
		y_train = np.concatenate((y_train , y_tr))


#Concatenating testing subjects 12 to 16
for index,i in enumerate(range(Start_TestSubject, End_TestSubject+1)):
	#Initialize the first loading
	if index == 0:
		test_data = scipy.io.loadmat('./data/train_subject' + str(i).zfill(2) +'.mat')
		x_test = test_data['X']
		if not KaggleTesting:
			y_test = test_data['y']  #Remember real test set data doesn't have this

		#Merge the Id's of the testing subjects for real Kaggle Testing
		else: 
			y_id = test_data['Id']

	else:
		test_data = scipy.io.loadmat('./data/train_subject' + str(i).zfill(2) +'.mat')
		x_tst= test_data['X']
		x_test = np.concatenate((x_test , x_tst))   #Add more test examples

		if not KaggleTesting:
			y_tst = test_data['y']
			y_test = np.concatenate((y_test , y_tst))

		else:
			ids = test_data['Id']
			y_id = np.concatenate((y_id , ids))


#To stop giving that error
x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)


print ("Training Set Shape: ", x_train.shape)
print ("Number of training examples: ", len(x_train))
print ("First training example shape: ", x_train[0].shape)
print (x_train[0:1])
print ("Test Set shape: ", x_test.shape)

#Create one-hot encoding labels 
y_train = keras.utils.to_categorical(y_train, 2)
if not KaggleTesting:
	y_test = keras.utils.to_categorical(y_test, 2)
print ("y train labels shape: ", y_train[0:1] ,y_train.shape)

#Delete the intial useless 0.5 seconds of sampling along the vertical axis (the first 125)
x_train = np.delete(x_train, range(0,125), axis=2)
x_test = np.delete(x_test, range(0,125), axis=2)
# x_train = np.delete(x_train, range(200,250), axis=2)
# x_test = np.delete(x_test, range(200,250), axis=2)

# random_seed = random.randint(2, 2000)
# np.random.seed(random_seed)
# np.random.shuffle(x_train)
# np.random.shuffle(x_train)

# np.random.seed(random_seed) #Reset the seed 
# np.random.shuffle(y_train)
# np.random.shuffle(y_train)

# Don't shuffle test data
# perm = np.random.permutation(x_train.shape[0])
# x_train = x_train[perm]
# y_train = y_train[perm]


#From 306 features to 150 (Dimensionality reduction)
# pca = PCA(n_components=250) 
# xx_train = np.zeros( (x_train.shape[0], 250, x_train.shape[2]) )
# xx_test = np.zeros( (x_test.shape[0], 250, x_test.shape[2]) )

# Normalize small values of training and test data
for i in range(0, len(x_train)):
	x_train[i] = preprocessing.scale(x_train[i])
	# xx_train[i]= pca.fit_transform(x_train[i].T).T

for i in range(0, len(x_test)):
	x_test[i] = preprocessing.scale(x_test[i])
	# xx_test[i]= pca.fit_transform(x_test[i].T).T

# x_train = xx_train
# x_test = xx_test


max_xtrain = np.amax(x_train)
min_xtrain = np.amin(x_train)
max_xtest = np.amax(x_test)
min_xtest = np.amin(x_test)
max_val_train = max(max_xtrain, np.abs(min_xtrain))
max_val_test = max(max_xtest, np.abs(min_xtest))
print ("Maximum & Minimum in new scaled X_train: ", max_xtrain, " ", min_xtrain)
x_train /= max_val_train
x_test /= max_val_test


#Mean of each of the 306 features along all the timesteps
# x_train_mean_vector = x_train.mean(axis=tuple([0,2]))
# x_test_mean_vector = x_test.mean(axis=tuple([0,2]))

# #The maximum Value
# x_train_max = np.amax(np.abs(x_train), axis=tuple([0,2]))
# x_test_max = np.amax(np.abs(x_test), axis=tuple([0,2]))

# #Normalize the values with mean 0 and std dev 1
# x_train = ( x_train - x_train_mean_vector.reshape((x_train.shape[1],1)) ) / x_train_max.reshape((x_train.shape[1],1))
# x_test = ( x_test - x_test_mean_vector.reshape((x_test.shape[1],1)) ) / x_test_max.reshape((x_test.shape[1],1))



# x_train = np.reshape(x_train, (-1, dim*win_len))
# input_dim = Input(shape = (x_train.shape[1]*x_train.shape[2], ))
# # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
# encoding_dim = 150
# # DEFINE THE ENCODER LAYERS
# encoded1 = Dense(512, activation = 'relu')(input_dim)
# encoded2 = Dense(10, activation = 'relu')(encoded1)
# encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)
# # DEFINE THE DECODER LAYERS
# decoded1 = Dense(5, activation = 'relu')(encoded3)
# decoded2 = Dense(10, activation = 'relu')(decoded1)
# decoded3 = Dense(ncol, activation = 'sigmoid')(decoded2)
# # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
# autoencoder = Model(input = input_dim, output = decoded3)
# # CONFIGURE AND TRAIN THE AUTOENCODER
# autoencoder.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy')
# autoencoder.fit(x_train, x_train, nb_epoch = 50, batch_size = 128, shuffle = True, validation_split = 0.2)
# # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
# encoder = Model(input = input_dim, output = encoded3)
# encoded_input = Input(shape = (encoding_dim, ))
# encoded_out = encoder.predict(X_test)
# encoded_out[0:2]



print ("New Xtrain + New Shape: ", x_train.shape , x_train[0:1])

#580 x 306 (num_features) x 375 (sampling freq of 250Hz for 1.5seconds)
_, dim, win_len = x_train.shape


def data_shaping(x_train, x_test, network_type):
	if network_type == "MLP":
		x_train = np.reshape(x_train, (-1, dim*win_len))
		x_test = np.reshape(x_test, (-1, dim*win_len))

	#LSTM expects 3D data (batch_size, timesteps, features)
	if network_type == "LSTM":
		x_train = np.swapaxes(x_train,1,2)
		x_test = np.swapaxes(x_test,1,2)

	if network_type == 'CNN' or network_type=='ConvLSTM':
		x_train = np.reshape(x_train, (-1, dim, win_len, 1))
		x_test = np.reshape(x_test, (-1, dim, win_len, 1))

	return x_train, x_test


def model_MLP(model, num_hidden_units):
	model.add(Dense(num_hidden_units, activation='relu', input_shape=(dim*win_len,), kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))

	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))
	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))
	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))
	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))


	# model.add(ConcreteDropout(Dense(2000, activation='relu', kernel_initializer='he_uniform'), input_shape=(dim*win_len,)))
	# model.add(ConcreteDropout(Dense(50, activation='relu', kernel_initializer='he_uniform') ))
	# model.add(ConcreteDropout(Dense(2000, activation='relu', kernel_initializer='he_uniform') ))
	# model.add(ConcreteDropout(Dense(306, activation='relu', kernel_initializer='he_uniform') ))


	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'), input_shape=(dim*win_len,)))
	# # model.add(BatchNormalization())
	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))
	# # model.add(BatchNormalization())
	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))


#return_sequence=TRUE, the output is sequence of same length (outputs after every input), 
#with return_sequence=FALSE, the output will be just one vector (outputs only after seeing everything)
def model_LSTM(model, num_hidden_LSTM_units):
	model.add(Bidirectional(LSTM(num_hidden_LSTM_units, return_sequences=True), input_shape=(win_len, dim)))
	model.add(Bidirectional(LSTM(num_hidden_LSTM_units, return_sequences=True)))
	model.add(Bidirectional(LSTM(num_hidden_LSTM_units, return_sequences=True)))##
	model.add(Bidirectional(LSTM(num_hidden_LSTM_units, return_sequences=False)))
	# model.add(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))
	# model.add(Dense(num_hidden_units, activation='relu',kernel_initializer='he_uniform'))
	# model.add(Dropout(0.5))
	model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))
	model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))


def model_CNN(model, num_feat_map):
	model.add(Conv2D(num_feat_map, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(dim, win_len, 1),
                 kernel_initializer='he_uniform',
                 padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(num_feat_map, kernel_size=(5, 5),
                 activation='relu',
                 kernel_initializer='he_uniform',
                 padding='same'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(num_feat_map, kernel_size=(5, 5),
                 activation='relu',
                 kernel_initializer='he_uniform',
                 padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))
	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))
	# model.add(ConcreteDropout(Dense(num_hidden_units, activation='relu', kernel_initializer='he_uniform') ))
	model.add(Dense(num_hidden_units, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dropout(0.4))
	model.add(Dense(num_hidden_units, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dropout(0.4))
	model.add(Dense(num_hidden_units, activation='relu',kernel_initializer='he_uniform'))
	model.add(Dropout(0.4))


def model_ConvLSTM(model, num_feat_map):
	# model.add(Conv2D(num_feat_map, kernel_size=(5, 5),
 #                 activation='relu',
 #                 input_shape=(dim, win_len, 1),
 #                 padding='same'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.5))
	# model.add(Permute((2, 1, 3))) # for swap-dimension
	# model.add(Reshape((-1,num_feat_map*dim)))
	# model.add(LSTM(32, return_sequences=False, stateful=False))
	# model.add(Dropout(0.5))

	model.add(ConvLSTM2D(num_feat_map,  kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu', input_shape=(dim, win_len, 1)))



def model_output(model):
	model.add(Dense(2, activation='softmax'))

 
network_type = 'MLP';  model_patience = 8;
# network_type = 'LSTM'; model_patience = 5;
# network_type = 'CNN';  model_patience = 40;     #25-30 epochs is enough
# network_type = 'ConvLSTM';  model_patience = 8;

###############Building the Model###############
num_hidden_units = 512
# num_hidden_LSTM_units = 64
num_hidden_LSTM_units = 128
# num_feat_map = 16
num_feat_map = 64
batch_size = 128
epochs = 60


x_train, x_test = data_shaping(x_train, x_test, network_type)

model = Sequential()
if network_type == "MLP":
	model_MLP(model, num_hidden_units)

if network_type == "LSTM":
	model_LSTM(model, num_hidden_LSTM_units)

if network_type == "CNN":
	model_CNN(model, num_feat_map)

if network_type == "ConvLSTM":
	model_ConvLSTM(model, num_feat_map)


model_output(model)

model.summary()
################################################

model.compile(
			  loss=keras.losses.categorical_crossentropy,
	 		  # loss=keras.losses.binary_crossentropy,
              optimizer='adam',
              # optimizer='rmsprop',
              metrics=['accuracy'])

H = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            # validation_data=(x_val, y_val)
            validation_split = 0.2,   #20% of last x_train and y_train taken for CV before shuffling
            callbacks =[EarlyStopping(monitor='val_loss', patience= model_patience)]  #6 epochs patience
            )

# model.fit(X train, Y train, batch size=batch size, epochs=num epochs, verbose=1, validation split=0.1, callbacks=[EarlyStopping(monitor=’val loss’, patience=7)])

if not KaggleTesting:
	score = model.evaluate(x_test, y_test, verbose=1)
	print('Test accuracy:', score[1])

predictions = model.predict(x_test)
if not KaggleTesting:
	print ("Test accuracy is: ", np.mean(np.array([np.argmax(i) for i in y_test]) == np.array([np.argmax(i) for i in predictions]) ))

print ("Output Predictions: ", predictions, len(predictions))
if not KaggleTesting:
	print ("Gold Results: ", y_test)


#y_id contains the test subject ids required for the submission
if KaggleTesting:
	rows = zip(np.array([i[0] for i in y_id]), np.array([np.argmax(i) for i in predictions]))
	with open("./KaggleSubmission.txt", "w") as f:
		writer = csv.writer(f)
		for row in rows:
			writer.writerow(row)


# import matplotlib.pyplot as plt
# summarize history for accuracy
# plt.plot(H.history['acc'])
# plt.plot(H.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(H.history['loss'])
# plt.plot(H.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()









