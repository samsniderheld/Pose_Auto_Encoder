import tensorflow as tf
import numpy as np
import glob
import random
import os
import cv2
import csv

class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for tf.keras'
    def __init__(self, args,shuffle=True,):
        self.dir = os.path.join(args.base_data_dir,args.input_data_dir_img + "*")
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.dims = args.img_width
        self.channels = args.channels
        self.files  = sorted(glob.glob(self.dir))
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.files))


    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_batches = int(np.floor(len(self.files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
      
        X = self.__data_generation(index)

        return X

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.files)

    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 
        
        files = self.files[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        
        X = np.empty((self.batch_size, self.dims, self.dims, self.channels))
        
        for i, file in enumerate(files):
          
            img = cv2.imread(files[i])

            img = cv2.resize(img, (self.dims, self.dims))

            img = img/255.
          
            X[i,] = img
        
        return X

class BoneDataGenerator(tf.keras.utils.Sequence):
#     'Generates data for tf.keras'
    def __init__(self, args, shuffle=True,):
        self.shuffle = shuffle
        self.img_dir = os.path.join(args.base_data_dir,args.input_data_dir_img + "*")
        self.csv_dir = os.path.join(args.base_data_dir,args.input_data_dir_csv+ "*")
        self.batch_size = args.batch_size
        self.dims = args.img_width
        self.csv_dims = args.csv_dims
        self.channels = args.channels
        self.img_files  = sorted(glob.glob(self.img_dir))
        self.csv_files  = sorted(glob.glob(self.csv_dir))
        self.all_files = list(zip(self.img_files,self.csv_files))
        self.on_epoch_end()
        self.count = self.__len__()
        print("number of all samples = ", len(self.img_files))


    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_batches = int(np.floor(len(self.all_files) / self.batch_size))
        return self.num_batches

    def __getitem__(self, index):
      
        X = self.__data_generation(index)

        return X

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.all_files)


    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 

        batch_files = self.all_files[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        
        X_img = np.empty((self.batch_size, self.dims, self.dims, self.channels))
        X_csv = np.empty((self.batch_size,52,3))

        # read image
        for i, batch_file in enumerate(batch_files):

            img = cv2.imread(batch_file[0])

            img = cv2.resize(img, (self.dims, self.dims))

            img = img/255.
          
            X_img[i,] = img

            all_values = []

            with open(batch_file[1]) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j,row in enumerate(csv_reader):
                    for k, val in enumerate(row[4:]):
                      X_csv[i,j,k] = float(val)/360.
          
            
            
        
        return ([X_img, X_csv],)