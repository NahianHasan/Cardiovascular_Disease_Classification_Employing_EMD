from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten,Add
from keras.layers import ZeroPadding1D,Input,AveragePooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras import regularizers
from keras import layers
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import SGD
import config
global C

def model_VGG19(data_length,number_of_classes,optimizer='rmsprop'):
	input_shape = (data_length,1)
	model = Sequential()

	model.add(ZeroPadding1D(1,input_shape=input_shape))
	model.add(Convolution1D(64, 3, activation='relu',strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(64, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(128, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(128, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(number_of_classes, activation='softmax'))
	model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
	#plot_model(model, to_file='./Outputs/Model_Figures/VGG19_model.png')
	return model

def model_Alexnet_single_channel(data_length,number_of_classes,optimizer='rmsprop'):
	input_shape = (data_length,1)
	model = Sequential()

	model.add(Convolution1D(96,11,input_shape=input_shape, strides=4, border_mode='valid'))#strides is same as 'strides'
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid' ))

	model.add(ZeroPadding1D(2))
	model.add(Convolution1D(256,5, strides=1))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(384,3, strides=1))
	model.add(Activation('relu'))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(384,3, strides=1))
	model.add(Activation('relu'))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(256,3, strides=1))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(number_of_classes,activation= 'softmax'))
	#plot_model(model, to_file='./Outputs/Model_Figures/Alexnet.png')
	return model

def model_ZFnet(data_length,number_of_classes,optimizer='rmsprop'):
	input_shape = (data_length,1)
	model = Sequential()

	model.add(Convolution1D(96,7,input_shape=input_shape, strides=4, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid' ))

	model.add(ZeroPadding1D(2))
	model.add(Convolution1D(256,5, strides=1))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid' ))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512,3, strides=1))
	model.add(Activation('relu'))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(1024,3, strides=1))
	model.add(Activation('relu'))

	model.add(ZeroPadding1D(1))
	model.add(Convolution1D(512,3, strides=1))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid' ))

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(number_of_classes,activation= 'softmax'))
	model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
	#plot_model(model, to_file='./Outputs/Model_Figures/ZFNet_model.png')
	return model
##########################################################################################
def identity_block(input_tensor, kernel_size, filters, stage, block):

	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 2
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution1D(filters1, 1, name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution1D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution1D(filters3, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)

	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):

	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 2
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution1D(filters1, 1, strides=strides,name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution1D(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution1D(filters3, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Convolution1D(filters3, 1, strides=strides,name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)

	return x

def ResNet50(data_length,number_of_classes,optimizer='sgd',
			include_top=True,
			weights='imagenet',
			pooling=None,
			input_tensor=None):

	input_shape = (data_length,1)
	bn_axis = 2

	input = Input(input_shape)
	x = Convolution1D(64, 7, strides=2, padding='same', name='conv1')(input)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(3, strides=2)(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	x = AveragePooling1D(7, name='avg_pool')(x)

	if include_top:
		x = Flatten()(x)
		x = Dense(number_of_classes, activation='softmax', name='ECG-8')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling1D()(x)
		elif pooling == 'max':
			x = GlobalMaxPooling1D()(x)

	model = Model(input, x, name='resnet50')
	#plot_model(model, to_file='./Outputs/Model_Figures/RESNet50_model.png')

	return model


###############  GooGleNet Model  ####################################


def inception_module(x, params, concat_axis,name,subsample=1, activation='relu',border_mode='same', weight_decay=None):

	(branch1, branch2, branch3, branch4) = params

	if weight_decay:
		W_regularizer = regularizers.l2(weight_decay)
		b_regularizer = regularizers.l2(weight_decay)
	else:
		W_regularizer = None
		b_regularizer = None

	pathway1 = Convolution1D(branch1[0], 1,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv1')(x)

	pathway2 = Convolution1D(branch2[0], 1,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv21')(x)
	pathway2 = Convolution1D(branch2[1], 3,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv22')(pathway2)

	pathway3 = Convolution1D(branch3[0], 1,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv31')(x)
	pathway3 = Convolution1D(branch3[1], 5,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv32')(pathway3)

	pathway4 = MaxPooling1D(pool_size=1, name=name+'pool1')(x)
	pathway4 = Convolution1D(branch4[0], 1,
							strides=subsample,
							activation=activation,
							border_mode=border_mode,
							W_regularizer=W_regularizer,
							b_regularizer=b_regularizer,
							bias=False,
							name=name+'conv4')(pathway4)

	return merge([pathway1, pathway2, pathway3, pathway4],
					mode='concat', concat_axis=concat_axis)

def conv_layer(x,nb_row,nb_filter,name,subsample=1,padding=None, activation='relu',border_mode='same', weight_decay=None):

	if weight_decay:
		W_regularizer = regularizers.l2(weight_decay)
		b_regularizer = regularizers.l2(weight_decay)
	else:
		W_regularizer = None
		b_regularizer = None

	x = Convolution1D(nb_filter, nb_row,
						strides=subsample,
						activation=activation,
						border_mode=border_mode,
						W_regularizer=W_regularizer,
						b_regularizer=b_regularizer,
						bias=False,
						name=name+'_Conv1')(x)

	if padding:
		for i in range(padding):
			x = ZeroPadding1D(padding=1, name=name+'_zp1_'+str(i))(x)

	return x

def model_GoogleNet(data_length,
					number_of_classes,name='Inception',
					LEARNING_RATE = 0.01,
					DROPOUT=0.4,
					weight_decay=0.0005,
					use_bn=True):
	
	input_shape = (data_length,1)
	input = Input(input_shape)
	CONCAT_AXIS = -1

	x = conv_layer(nb_filter=64,nb_row=7,  padding=3,name=name+'conv_layer1')(input)
	x = MaxPooling1D(strides=3, pool_size=2, name=name+'pool1')(x)

	x = conv_layer(x, nb_row=1, nb_filter=64, name=name+'conv_layer2')
	x = conv_layer(x, nb_row=1, nb_filter=192,  padding=1,name=name+'conv_layer3')
	x = MaxPooling1D(strides=3, pool_size=2, name=name+'pool2')(x)

	x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )], concat_axis=CONCAT_AXIS,name=name+'inception1')
	x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )], concat_axis=CONCAT_AXIS,name=name+'inception2')

	x = MaxPooling1D(strides=1, pool_size=1, name=name+'pool3')(x)
	x = ZeroPadding1D(padding=1, name=name+'zp1')(x)

	x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )], concat_axis=CONCAT_AXIS,name=name+'inception3')
	x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )], concat_axis=CONCAT_AXIS,name=name+'inception4')
	x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )], concat_axis=CONCAT_AXIS,name=name+'inception5')
	x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )], concat_axis=CONCAT_AXIS,name=name+'inception6')

	x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], 
								concat_axis=CONCAT_AXIS,name=name+'inception7')
	x = MaxPooling1D(strides=1, pool_size=1, name=name+'pool4')(x)
	x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)], 
								concat_axis=CONCAT_AXIS,name=name+'inception8')
	x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)], 
								concat_axis=CONCAT_AXIS,name=name+'inception9')

	x = AveragePooling1D(strides=1, name=name+'pool5')(x)
	x = Flatten(name=name+'Flat1')(x)
	x = Dropout(DROPOUT,name=name+'DP1')(x)
	x = Dense(output_dim=number_of_classes,activation='linear',name=name+'Dense1')(x)
	x = Dense(output_dim=number_of_classes,activation='softmax',name=name+'Dense2')(x)

	model = Model(input, x, name='InceptionV1')
	#plot_model(model, to_file='./Outputs/Model_Figures/Inception_model.png')

	return model


##########################################################################################
def model_VGG16(data_length,number_of_classes,optimizer='rmsprop'):
	input_shape = (data_length,1)
	model = Sequential()

	model.add(ZeroPadding1D(padding=1,input_shape=input_shape))
	model.add(Convolution1D(64, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(64, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(128, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(128, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(256, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(256, 3, activation='relu'))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(ZeroPadding1D(padding=1))
	model.add(Convolution1D(512, 3, activation='relu', strides=1))
	model.add(MaxPooling1D(2, strides=2))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(number_of_classes, activation='softmax'))
	model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
	#plot_model(model, to_file='./Outputs/Model_Figures/VGG16_model.png')
	return model
##################################################################################
def Custom_architecture(self,init='glorot_uniform',optimizer='rmsprop'):
	model = Sequential()
	model.add(Dense(self.samplenumber,input_dim = self.samplenumber, init = 'uniform', activation = 'relu',
					W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(self.samplenumber/2,init = init, activation = 'relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(self.samplenumber/4,init = init, activation = 'relu', W_constraint=maxnorm(3)))
	model.add(Dense(self.classes,init = 'uniform', activation = 'softmax'))

	if optimizer=='sgd':
		sgd = SGD(lr=0.000001, momentum=0.9, decay=0.0, nesterov=False)
		#Compile Model
		model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	else:
		model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
	#plot_model(model, to_file='./Outputs/Model_Figures/Custom_Model.png')
	return model

##################################################################################
class MODELS:
	def __init__(self):
		C = config.Config()
		self.samplenumber = C.samplenumber
		self.optimizer = C.optimizer
		self.classes = C.classes
		self.IMF_models={'1':self.IMF_model_1,'2':self.IMF_model_2,'3':self.IMF_model_3,'4':self.IMF_model_4,'5':self.IMF_model_5,'6':self.IMF_model_6}
		self.data_length = self.samplenumber
		if C.architect == 'RESNet50':
			self.model = ResNet50(self.data_length,self.classes,self.optimizer)
		elif C.architect == 'AlexNet':
			self.model = model_Alexnet_single_channel(self.data_length,self.classes,self.optimizer)
		elif C.architect == 'VGGNet16':
			self.model = model_VGG16(data_length,number_of_classes,self.optimizer)
		elif C.architect == 'VGGNet19':
			self.model = model_VGG19(self.data_length,self.classes,self.optimizer)
		elif C.architect == 'ZFNet':
			self.model = model_ZFnet(self.data_length,self.classes,self.optimizer)
		elif C.architect == 'Custom':
			self.model = Custom_architecture(self)
		elif C.architect == 'Inception':
			self.model = model_GoogleNet(self.data_length,self.classes)

	def IMF_model_1(self):
		return self.model
	def IMF_model_2(self):
		return self.model
	def IMF_model_3(self):
		return self.model
	def IMF_model_4(self):
		return self.model
	def IMF_model_5(self):
		return self.model
	def IMF_model_6(self):
		return self.model
