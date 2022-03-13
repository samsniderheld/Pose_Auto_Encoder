from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.math import multiply

#via https://keras.io/examples/variational_autoencoder/
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def create_variational_bone_auto_encoder(latent_dim = 10, dims = 128, kernal_size = 3, beta = 1):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,3)

    #image encoder input
    image_encoder_input = Input(shape=input_shape)

    #bone encoder input
    bone_encoder_input = Input(shape=(52,3))

    bone_weight_input = Input(shape=(52,3))


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

    z_mean = Dense(latent_size)(x)
    z_log_var = Dense(latent_size)(x)

    #z layer layer
    z = Lambda(sampling, output_shape=(latent_size,), name='z')([z_mean,z_log_var])

    #instantiate encoder model
    encoder = Model(image_encoder_input, [z_mean, z_log_var, z], name='image_encoder')
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
    bone_decoder_output = Conv2DTranspose(3, (kernal_size, kernal_size), strides = (1,1), activation='sigmoid', padding='same')(x2)

    # instantiate bone decoder model
    bone_decoder = Model(bone_decoder_input, bone_decoder_output, name='bone_decoder')
    bone_decoder.summary()


    bone_variational_auto_encoder_output = bone_decoder(encoder(image_encoder_input)[2])

    bone_variational_auto_encoder = Model([image_encoder_input, bone_encoder_input, bone_weight_input], 
        bone_variational_auto_encoder_output, name='bone_variational_auto_encoder')

    bone_variational_auto_encoder.summary()

    #define losses
    bone_reconstruction_loss = mse(K.flatten(multiply(bone_encoder_input,bone_weight_input)), 
        K.flatten(bone_variational_auto_encoder_output))
    
    # bone_reconstruction_loss *= 52 * 3
    kl_loss = (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)) * beta
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(bone_reconstruction_loss + kl_loss)

    bone_variational_auto_encoder.add_loss(bone_reconstruction_loss)

    bone_variational_auto_encoder.compile(optimizer='adam')

    return encoder, bone_decoder, bone_variational_auto_encoder


                 