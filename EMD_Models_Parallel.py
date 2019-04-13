from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import Model
from keras.layers import TimeDistributed,SimpleRNN,LSTM,GRU,Bidirectional,CuDNNLSTM
from keras.layers import ZeroPadding1D,Input,AveragePooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import Dense,merge,concatenate,Flatten,Dropout,Add
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras import regularizers
from keras import layers
from keras.utils import plot_model
from keras import backend as K
from keras.layers import LSTM
import config
global C



def model_VGG19(input_layer,data_length,number_of_classes,name):
	input_shape = (data_length,1)

	x = ZeroPadding1D(1,input_shape=input_shape,name=name+'zp0')(input_layer)
	x = Convolution1D(64, 3, activation='relu',strides=1,name=name+'conv1')(x)
	x = ZeroPadding1D(1,name=name+'zp1')(x)
	x = Convolution1D(64, 3, activation='relu', strides=1,name=name+'conv2')(x)
	x = MaxPooling1D(2, strides=2,name=name+'pool1')(x)
	'''
	x = ZeroPadding1D(1,name=name+'zp2')(x)
	x = Convolution1D(128, 3, activation='relu', strides=1,name=name+'conv3')(x)
	x = ZeroPadding1D(1,name=name+'zp3')(x)
	x = Convolution1D(128, 3, activation='relu', strides=1,name=name+'conv4')(x)
	x = MaxPooling1D(2, strides=2,name=name+'pool2')(x)

	x = ZeroPadding1D(1,name=name+'zp4')(x)
	x = Convolution1D(256, 3, activation='relu', strides=1,name=name+'conv5')(x)
	x = ZeroPadding1D(1,name=name+'zp5')(x)
	x = Convolution1D(256, 3, activation='relu', strides=1,name=name+'conv6')(x)

	x = ZeroPadding1D(1,name=name+'zp6')(x)
	x = Convolution1D(256, 3, activation='relu', strides=1,name=name+'conv7')(x)
	x = ZeroPadding1D(1,name=name+'zp7')(x)
	x = Convolution1D(256, 3, activation='relu', strides=1,name=name+'conv8')(x)
	x = MaxPooling1D(2, strides=2,name=name+'pool3')(x)

	x = ZeroPadding1D(1,name=name+'zp8')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv9')(x)
	x = ZeroPadding1D(1,name=name+'zp9')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv10')(x)

	x = ZeroPadding1D(1,name=name+'zp10')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv11')(x)
	x = ZeroPadding1D(1,name=name+'zp11')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv12')(x)
	x = MaxPooling1D(2, strides=2,name=name+'pool4')(x)

	x = ZeroPadding1D(1,name=name+'zp12')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv13')(x)
	x = ZeroPadding1D(1,name=name+'zp13')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv14')(x)

	x = ZeroPadding1D(1,name=name+'zp14')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv15')(x)
	x = ZeroPadding1D(1,name=name+'zp15')(x)
	x = Convolution1D(512, 3, activation='relu', strides=1,name=name+'conv16')(x)
	x = MaxPooling1D(2, strides=2,name=name+'pool5')(x)
	'''

	
	#x = TimeDistributed(Flatten(name=name+'flat'))(x)
	#x = Bidirectional(LSTM(500,return_sequences = True,name=name+'LSTM1'))(x)
	x = CuDNNLSTM(20,name=name+'GRU1')(x)
	x = Dropout(0.5,name=name+'dr1')(x)
	x = Dense(4096, activation='relu',name=name+'dn1')(x)
	#x = Bidirectional(LSTM(500,return_sequences = True,name=name+'LSTM2'))(x)
	x = CuDNNLSTM(20,name=name+'GRU2')(x)
	x = Dropout(0.5,name=name+'dr2')(x)
	x = Dense(2048, activation='relu',name=name+'dn2')(x)
	#x = Bidirectional(LSTM(250,name=name+'LSTM3'))(x)
	x = CuDNNLSTM(20,name=name+'GRU3')(x)
	x = Dense(number_of_classes, activation='relu',name=name+'Final_Layer')(x)
	#plot_model(model, to_file='./Outputs/Model_Figures/VGG19_model.png')

	return x

def model_Alexnet_single_channel(input_layer,data_length,number_of_classes,name):
	input_shape = (data_length,1)
	
	x = Convolution1D(96,11,strides=4, border_mode='valid',name=name+'conv1')(input_layer)#strides is same as 'strides'
	x = Activation('relu',name=name+'act1')(x)
	x = BatchNormalization(name=name+'bn1')(x)
	x = MaxPooling1D(pool_size=2, strides=2, padding='valid',name=name+'pool1' )(x)

	x = ZeroPadding1D(2,name=name+'zp1')(x)
	x = Convolution1D(256,5, strides=1,name=name+'conv2')(x)
	x = Activation('relu',name=name+'act2')(x)
	x = BatchNormalization(name=name+'bn2')(x)
	x = MaxPooling1D(pool_size=3, strides=2, padding='valid',name=name+'pool2')(x)

	x = ZeroPadding1D(1,name=name+'zp2')(x)
	x = Convolution1D(384,3, strides=1,name=name+'conv3')(x)
	x = Activation('relu',name=name+'act3')(x)

	x = ZeroPadding1D(1,name=name+'zp3')(x)
	x = Convolution1D(384,3, strides=1,name=name+'conv4')(x)
	x = Activation('relu',name=name+'act4')(x)

	x = ZeroPadding1D(1,name=name+'zp4')(x)
	x = Convolution1D(256,3, strides=1,name=name+'conv5')(x)
	x = Activation('relu',name=name+'act5')(x)
	x = MaxPooling1D(pool_size=3, strides=2, padding='valid',name=name+'pool3')(x)

	x = Flatten(name=name+'flat')(x)
	x = Dense(4096,name=name+'dn1')(x)
	x = Activation('relu',name=name+'act6')(x)
	x = Dropout(0.5,name=name+'dr1')(x)

	x = Dense(4096,name=name+'dn2')(x)
	x = Activation('relu',name=name+'act7')(x)
	x = Dropout(0.5,name=name+'dr2')(x)

	x = Dense(number_of_classes,activation= 'relu',name=name+'softmax')(x)
	return x
	
	
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

def model_GoogleNet(input_layer,data_length,
					number_of_classes,name,
					LEARNING_RATE = 0.01,
					DROPOUT=0.4,
					weight_decay=0.0005,
					use_bn=True):

	CONCAT_AXIS = -1

	x = conv_layer(input_layer,nb_filter=64,nb_row=7,  padding=3,name=name+'conv_layer1')
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

	return x



###############   RESNET50   Architecture   ########################
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

def ResNet50(input_layer,data_length,number_of_classes,name,
			include_top=True,
			weights='imagenet',
			pooling=None,
			input_tensor=None):

	bn_axis = 2
	x = Convolution1D(64, 7, strides=2, padding='same')(input_layer)
	x = BatchNormalization(axis=bn_axis, name=name+'bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(3, strides=2)(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block=name+'a', strides=1)
	x = identity_block(x, 3, [64, 64, 256], stage=2, block=name+'b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block=name+'c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block=name+'a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block=name+'b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block=name+'c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block=name+'d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block=name+'a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block=name+'b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block=name+'c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block=name+'d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block=name+'e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block=name+'f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block=name+'a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block=name+'b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block=name+'c')

	x = AveragePooling1D(7, name=name+'avg_pool')(x)

	if include_top:
		x = Flatten()(x)
		#x = Dense(number_of_classes, activation='relu', name=name+'individual_final')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling1D()(x)
		elif pooling == 'max':
			x = GlobalMaxPooling1D()(x)

	return x

########################################################################################

class MODELS:
	def __init__(self):
		C = config.Config()
		self.samplenumber = C.samplenumber
		self.optimizer = C.optimizer
		self.classes = C.classes
		self.data_length = self.samplenumber

	def EMD_Parallel_Models(self):
		C = config.Config()
		input_shape = (self.data_length,1)
		x1 = Input(input_shape)
		x2 = Input(input_shape)
		x3 = Input(input_shape)
		#x4 = Input(input_shape)
		#x5 = Input(input_shape)
		#x6 = Input(input_shape)
		if C.architect == 'RESNet50':
			#channel 1
			y1 = ResNet50(x1,self.data_length,self.classes,name='IMF1_')
			#channel 2
			y2 = ResNet50(x2,self.data_length,self.classes,name='IMF2_')
			#channel 3
			y3 = ResNet50(x3,self.data_length,self.classes,name='IMF3_')
			#channel 4
			y4 = ResNet50(x4,self.data_length,self.classes,name='IMF4_')
			#channel 5
			y5 = ResNet50(x5,self.data_length,self.classes,name='IMF5_')
			#channel 6
			y6 = ResNet50(x6,self.data_length,self.classes,name='IMF6_')
		elif C.architect == 'AlexNet':
			y1 = model_Alexnet_single_channel(x1,self.data_length,self.classes,name='IMF1_')
			#channel 
			y2 = model_Alexnet_single_channel(x2,self.data_length,self.classes,name='IMF2_')
			#channel 3
			y3 = model_Alexnet_single_channel(x3,self.data_length,self.classes,name='IMF3_')
			#channel 4
                        #y4 = model_Alexnet_single_channel(x4,self.data_length,self.classes,name='IMF4_')
                        #channel 5
                        #y5 = model_Alexnet_single_channel(x5,self.data_length,self.classes,name='IMF5_')
                        #channel 6
                        #y6 = model_Alexnet_single_channel(x6,self.data_length,self.classes,name='IMF6_')
		elif C.architect == 'VGGNet19':
			y1 = model_VGG19(x1,self.data_length,self.classes,name='IMF1_')
			#channel 
			y2 = model_VGG19(x2,self.data_length,self.classes,name='IMF2_')
			#channel 3
			y3 = model_VGG19(x3,self.data_length,self.classes,name='IMF3_')
			#channel 4
                        y4 =  model_VGG19(x4,self.data_length,self.classes,name='IMF4_')
                        #channel 5
                        y5 =  model_VGG19(x5,self.data_length,self.classes,name='IMF5_')
                        #channel 6
                        y6 =  model_VGG19(x6,self.data_length,self.classes,name='IMF6_')
		elif C.architect == 'Inception':
			y1 = model_GoogleNet(x1,self.data_length,self.classes,name='IMF1_')
			#channel 
			y2 = model_GoogleNet(x2,self.data_length,self.classes,name='IMF2_')
			#channel 3
			y3 = model_GoogleNet(x3,self.data_length,self.classes,name='IMF3_')
			#channel 4
                        y4 =  model_GoogleNet(x4,self.data_length,self.classes,name='IMF4_')
                        #channel 5
                        y5 =  model_GoogleNet(x5,self.data_length,self.classes,name='IMF5_')
                        #channel 6
                        y6 =  model_GoogleNet(x6,self.data_length,self.classes,name='IMF6_')
		print 'Y returned'
		#y = concatenate([y1,y2,y3,y4,y5,y6],axis=-1,name = 'concat2')#Concatenated final 6 IMF softmax outputs = 48 neurons
		y = concatenate([y1,y2,y3],axis=-1,name = 'concat2')#Concatenated final 6 IMF softmax outputs = 48 neurons

		y = Dense(24,init = 'uniform', activation = 'relu',W_constraint=maxnorm(3), name='Concatenated_Dense_1')(y)
		y = Dropout(0.2,name='Concatenated_Dropout_1')(y)
		y = Dense(16,init = 'uniform', activation = 'relu', W_constraint=maxnorm(3),name='Concatenated_Dense_2')(y)
		y = Dropout(0.2,name='Concatenated_Dropout_2')(y)
		y = Dense(self.classes,init = 'uniform', activation = 'softmax',name='Final_softmax_layer')(y)

		#model = Model(inputs=[x1,x2,x3,x4,x5,x6], outputs=y,name=C.architect+'_Parallel')
		model = Model(inputs=[x1,x2,x3], outputs=y,name=C.architect+'_Parallel')
		print 'Model created'
		#plot_model(model, to_file='./Outputs/Model_Figures/'+C.architect+'_Parallel',show_shapes=True, show_layer_names=True)

		return model
