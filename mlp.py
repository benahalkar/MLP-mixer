# import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LayerNormalization, Input, Add, Conv2D, Reshape, GlobalAveragePooling1D, Dropout, Flatten
from tensorflow.keras.activations import gelu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16

# def tokenmixingMLP(x, hidden_neurons):
#   out = Dense(hidden_neurons)(x)
#   out = gelu(out)
#   out = Dense(x.shape[ -1 ])(out)
#   return out


# def channelmixingMLP(x, hidden_neurons):
#   out = Dense(hidden_neurons)(x)
#   out = gelu(out)
#   out = Dense(x.shape[ -1 ])(out)
#   return out


# create one block of the MLP-mixer architecture
def MLPmixer(x, tokenMLP_size, channelMLP_size, dropout):
  # add LayerNormalization layer  
  y = LayerNormalization(
        axis=-1,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        )(x)
    
  # transpose image 
  y = tf.transpose(y, perm=[0, 2, 1])

  # token mixing MLP
  a = y
  # dense layer
  y = Dense(tokenMLP_size)(y)
  # GELU layer  
  y = gelu(y)
  # dense layer  
  y = Dense(a.shape[-1])(y)
  # dropout layer
  y = Dropout(dropout)(y)
    
  # transpose image 
  y = tf.transpose(y, perm=[0, 2, 1])
   
  # skip connection
  y_out = Add()([x, y])
    
  # add LayerNormalization layer  
  y = LayerNormalization(
        axis=-1,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        )(y_out)

  # chanel mixing MLP
  a = y
  # dense layer
  y = Dense(channelMLP_size)(y)
  # GELU layer  
  y = gelu(y)
  # dense layer  
  y = Dense(a.shape[-1])(y)
  # dropout layer
  y = Dropout(dropout)(y)
  
  # skip connection
  y = Add()([y_out, y])

  return y


# create MLP architecture
def makeModel(input_shape,
              number_of_mixers,
              token_mixing_num_mlps,
              channel_mixing_num_mlps,
              patch_size, hidden_dims,
              num_classes,
              dropout=0.1):
  
  # input layer
  input = Input(shape=(input_shape))
  
  # creating patches and reshaping the patches
  x = Conv2D(hidden_dims, kernel_size=patch_size, strides=patch_size)(input)
  x = Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
    
  # adding MLP-mixer layers
  for i in range(number_of_mixers):
    x = MLPmixer(x, token_mixing_num_mlps, channel_mixing_num_mlps, dropout)
  
  # Global average pooling layer
  x = GlobalAveragePooling1D()(x)
   
  # dense layer
  output = Dense(num_classes, activation = "softmax")(x)

  model = Model(input, output)
  return model
