import os
from Model.auto_encoder import create_auto_encoder
from Utils.reporting import *
from Data_Utils.generate_samples import *
from Data_Utils.data_generator import DataGenerator


def train(args):

    #setup data
    random_img = get_random_sample_img(args)
    random_img_samples = get_random_samples(args,100)

    data_generator = DataGenerator(args,shuffle=True)

    encoder,decoder,auto_encoder = create_auto_encoder(dims=args.img_width, latent_dim = args.latent_dim)

    for i,epoch in enumerate(range(args.num_epochs)):

        print(f'training epoch {str(i)}')

        auto_encoder.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        if(i%args.print_freq == 0):
            generate_test_image(auto_encoder,random_img,args.output_test_img_dir,i)
            generate_test_image_grid(auto_encoder,random_img_samples,args.output_test_img_dir,i)

        if(i%args.save_freq == 0):
            encoder_save_path = os.path.join(args.saved_model_dir,f"{i:04d}_encoder_model.h5")
            encoder.save(encoder_save_path)

            decoder_save_path = os.path.join(args.saved_model_dir,f"{i:04d}_decoder_model.h5")
            decoder.save(decoder_save_path)

    generate_test_image(auto_encoder,random_img,args.output_test_img_dir,args.num_epochs)
    generate_test_image_grid(auto_encoder,random_img_samples,args.output_test_img_dir,args.num_epochs)

    encoder_save_path = os.path.join(args.saved_model_dir,f"{args.num_epochs:04d}_encoder_model.h5")
    encoder.save(encoder_save_path)

    decoder_save_path = os.path.join(args.saved_model_dir,f"{args.num_epochs:04d}_decoder_model.h5")
    decoder.save(decoder_save_path)

