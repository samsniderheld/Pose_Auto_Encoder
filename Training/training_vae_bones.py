import os
from Model.bone_variational_auto_encoder import create_variational_bone_auto_encoder
from Utils.reporting import *
from Data_Utils.generate_samples import *
from Data_Utils.data_generator import BoneDataGenerator



def train_vae_bones(args):

    #setup data
    random_sample,random_csv_sample = get_random_sample_img(args)

    data_generator = BoneDataGenerator(args)

    encoder, bone_decoder, bone_auto_encoder = create_variational_bone_auto_encoder(
        dims=args.img_width, latent_dim = args.latent_dim)

    all_history = []
    lowest_loss = 10000

    output_test_csv_dir = os.path.join(args.base_results_dir,args.output_test_csv_dir)
    model_save_path = os.path.join(args.base_results_dir, args.saved_model_dir)
    output_history_path = os.path.join(args.base_results_dir,args.history_dir)

    for i,epoch in enumerate(range(args.num_epochs)):

        print(f'training epoch {str(i)}')

        history = bone_auto_encoder.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        loss = round(history.history['loss'][0],4)

        if(loss < lowest_loss):
            generate_bone_accuracy_table(bone_auto_encoder,random_sample,random_csv_sample, 
                output_test_csv_dir, i, args.print_csv, args.bone_training)

            encoder_save_path = os.path.join(model_save_path,f"{i:04d}_encoder_model.h5")
            encoder.save(encoder_save_path)

            bone_decoder_save_path = os.path.join(model_save_path,f"{i:04d}_bone_decoder_model.h5")
            bone_decoder.save(bone_decoder_save_path)

            lowest_loss = loss

        all_history.append(history.history['loss'][0])


    generate_bone_accuracy_table(bone_auto_encoder,random_sample,random_csv_sample, 
                output_history_path, args.num_epochs, args.print_csv, args.bone_training)

    encoder_save_path = os.path.join(model_save_path,f"{args.num_epochs:04d}_encoder_model.h5")
    encoder.save(encoder_save_path)


    bone_decoder_save_path = os.path.join(model_save_path,f"{args.num_epochs:04d}_bone_decoder_model.h5")
    bone_decoder.save(bone_decoder_save_path)

    save_experiment_history(args,all_history,output_history_path)

