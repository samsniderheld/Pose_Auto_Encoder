from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

def create_bone_auto_encoder(latent_dim = 10, dims = 128, kernal_size = 3):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,3)

    #image encoder input
    image_encoder_input = Input(shape=input_shape)

    #bone encoder input
    bone_encoder_input = Input(shape=(52,2))


    #downsampling/encoder
    x = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(image_encoder_input)
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
    encoder = Model (image_encoder_input,encoder_output, name="image_decoder")

    encoder.summary()


    #output dims = x,52,3

    bone_decoder_input = Input(shape=(latent_size,))

    x2 = Dense(13*latent_size)(bone_decoder_input)

    x2 = Reshape((1,13, latent_size))(x2)
    x2 = Conv2DTranspose(256, (kernal_size, kernal_size), strides = (1,2), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(256, (kernal_size, kernal_size), strides = (1,1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(128, (kernal_size, kernal_size), strides = (1,2), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2DTranspose(128, (kernal_size, kernal_size), strides = (1,1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    bone_decoder_output = Conv2DTranspose(2, (kernal_size, kernal_size), strides = (1,1), activation='sigmoid', padding='same')(x2)

    # instantiate bone decoder model
    bone_decoder = Model(bone_decoder_input, bone_decoder_output, name='bone_decoder')
    bone_decoder.summary()



    bone_auto_encoder_output = bone_decoder(encoder(image_encoder_input))

    bone_auto_encoder = Model([image_encoder_input,bone_encoder_input], bone_auto_encoder_output, name='bone_auto_encoder')

    bone_auto_encoder.summary()

    #define losses
    bone_reconstruction_loss = mse(K.flatten(bone_encoder_input), K.flatten(bone_auto_encoder_output))
    
    bone_auto_encoder.add_loss(bone_reconstruction_loss)

    bone_auto_encoder.compile(optimizer='adam')

    return encoder, bone_decoder, bone_auto_encoder


                 