import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def initial_block(inp):
    """
    Args:
        inp:
    Returns:
        concat:
    """
    inp1 = inp
    conv = Conv2D(filters=13, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(inp)
    pool = MaxPool2D(2)(inp1)
    concat = concatenate([conv, pool])
    return concat


def encoder_bottleneck(inp, filters, name, dilation_rate=2, downsample=False, dilated=False, asymmetric=False, drop_rate=0.1):
    """
    Args:
        inp:
        filters: 
        name: 
        dilation_rate:
        downsample: 
        dilated: 
        asymmetric: 
        drop_rate:
    Returns:
        concat:
    """

    reduce = filters // 4
    down = inp
    kernel_stride = 1

    #Downsample
    if downsample:
        kernel_stride = 2
        pad_activations = filters - inp.shape.as_list()[-1]
        down = MaxPool2D(2)(down)
        down = Permute(dims=(1, 3, 2))(down)
        down = ZeroPadding2D(padding=((0, 0), (0, pad_activations)))(down)
        down = Permute(dims=(1, 3, 2))(down)

    #1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=kernel_stride, strides=kernel_stride, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    #Conv
    if not dilated and not asymmetric:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same', kernel_initializer='he_normal', name=f'{name}_conv_reg')(x)
    elif dilated:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal', name=f'{name}_reduce_dilated')(x)
    elif asymmetric:
        x = Conv2D(filters=reduce, kernel_size=(1,5), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_asymmetric')(x)
        x = Conv2D(filters=reduce, kernel_size=(5,1), padding='same', kernel_initializer='he_normal', name=name)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    #1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = SpatialDropout2D(rate=drop_rate)(x)

    concat = Add()([x, down])
    concat = PReLU(shared_axes=[1, 2])(concat)
    return concat


def decoder_bottleneck(inp, filters, name, upsample=False):
    """
    Args:
        inp:
        filters: 
        name: 
        upsample: 

    Returns:
        concat:
    """
    reduce = filters // 4
    up = inp

    #Upsample
    if upsample:
        up = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_upsample')(up)
        up = UpSampling2D(size=2)(up)

    #1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    #Conv
    if not upsample:
        x = Conv2D(filters=reduce, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', name=f'{name}_conv_reg')(x)
    else:
        x = Conv2DTranspose(filters=reduce, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal', name=f'{name}_transpose')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    #1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)

    concat = Add()([x, up])
    concat = ReLU()(concat)

    return concat


def ENet(image_size, nclasses):
    """
    Args:
        image_size: size of the image as input
        nclasses: Total nb of classes

    Returns:
        model: Keras model in .h5 format
    """
    
    inp = Input(shape=(image_size, image_size, 3))
    enc = initial_block(inp)

    #Bottleneck 1.0
    enc = encoder_bottleneck(enc, 64, name='enc1', downsample=True, drop_rate=0.001)

    enc = encoder_bottleneck(enc, 64, name='enc1.1', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.2', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.3', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.4', drop_rate=0.001)

    #Bottleneck 2.0
    enc = encoder_bottleneck(enc, 128, name='enc2.0', downsample=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.1')
    enc = encoder_bottleneck(enc, 128, name='enc2.2', dilation_rate=2, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.3', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.4', dilation_rate=4, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.5')
    enc = encoder_bottleneck(enc, 128, name='enc2.6', dilation_rate=8, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.7', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.8', dilation_rate=16, dilated=True)

    #Bottleneck 3.0
    enc = encoder_bottleneck(enc, 128, name='enc3.0')
    enc = encoder_bottleneck(enc, 128, name='enc3.1', dilation_rate=2, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.2', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.3', dilation_rate=4, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.4')
    enc = encoder_bottleneck(enc, 128, name='enc3.5', dilation_rate=8, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.6', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.7', dilation_rate=16, dilated=True)

    #Bottleneck 4.0
    dec = decoder_bottleneck(enc, 64, name='dec4.0', upsample=True)
    dec = decoder_bottleneck(dec, 64, name='dec4.1')
    dec = decoder_bottleneck(dec, 64, name='dec4.2')

    #Bottleneck 5.0
    dec = decoder_bottleneck(dec, 16, name='dec5.0', upsample=True)
    dec = decoder_bottleneck(dec, 16, name='dec5.1')

    dec = Conv2DTranspose(filters=nclasses, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal', name='fullconv')(dec)
    dec = Activation('softmax')(dec)

    model = Model(inputs=inp, outputs=dec, name='Enet')
    return model