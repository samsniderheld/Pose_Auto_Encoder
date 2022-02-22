from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

def create_auto_encoder(latent_dim = 10, dims = 128, kernal_size = 3):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,3)

    #encoder input
    encoder_input = Input(shape=input_shape)

    #downsampling/encoder
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    shape = K.int_shape(x)

    x = Flatten()(x)

    encoder_output = Dense(latent_size)(x)

    # get shape info for later


    
    

    # instantiate encoder model
    encoder = Model (encoder_input,encoder_output, name="encoder")

    encoder.summary()

    

    decoder_input = Input(shape=(latent_size,))


    x2 = Dense(shape[1] * shape[2] * shape[3])(decoder_input)

    x2 = Reshape((shape[1], shape[2], shape[3]))(x2)

    #decoder layers
    x2 = Conv2D(256, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(256, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(8,(kernal_size, kernal_size), strides=(2, 2), padding='same')(x2)

    x2 = Conv2D(128, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(128, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(8,(kernal_size, kernal_size), strides=(2, 2), padding='same')(x2)

    x2 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(8, (kernal_size, kernal_size), strides=(2, 2), padding='same')(x2)

    #extra for 256 dims
    x2 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(8, (kernal_size, kernal_size), strides=(2, 2), padding='same')(x2)

    x2 = Conv2D(16, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(16, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(16, (kernal_size, kernal_size), strides=(2, 2), padding='same')(x2)

    decoder_output = Conv2D(3, (kernal_size, kernal_size), activation='sigmoid', padding='same')(x2)

    # instantiate decoder model
    decoder = Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()


    # instantiate VAE model
    auto_encoder_output = decoder(encoder(encoder_input))

    auto_encoder = Model(encoder_input, auto_encoder_output, name='auto_encoder')

    auto_encoder.summary()

    #define losses
    reconstruction_loss = mse(K.flatten(encoder_input), K.flatten(auto_encoder_output))
    
    auto_encoder.add_loss(reconstruction_loss)

    auto_encoder.compile(optimizer="adam")

    return encoder, decoder, auto_encoder


                 