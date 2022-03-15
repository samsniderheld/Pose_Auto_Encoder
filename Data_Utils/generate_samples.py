import tensorflow as tf
import numpy as np
import random
import glob
import os
import cv2
import csv
from Utils.util_functions import *

def get_random_sample_img(args):

    img_data_path = os.path.join(args.base_data_dir,args.input_data_dir_img + "*")

    img_paths = sorted(glob.glob(img_data_path),key=natural_keys)

    rand_idx = random.randint(0,len(img_paths))

    img = cv2.imread(img_paths[rand_idx])

    img = cv2.resize(img, (args.img_width,args.img_width))

    img = img/255.

    csv_data_path = os.path.join(args.base_data_dir,args.input_data_dir_csv + "*")

    csv_paths = sorted(glob.glob(csv_data_path),key=natural_keys)

    X_img = np.empty((1, args.img_width, args.img_width, args.channels))
    X_csv = np.empty((1, 52,2))

    X_img[0,] = img

    with open(csv_paths[rand_idx]) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j,row in enumerate(csv_reader):
                    for k, val in enumerate(row[1:3]):
                      X_csv[0,j,k] = float(val)/360.

    return ([X_img, X_csv],), X_csv


def get_random_sample_img_with_weight(args):

    img_data_path = os.path.join(args.base_data_dir,args.input_data_dir_img + "*")

    img_paths = sorted(glob.glob(img_data_path),key=natural_keys)

    rand_idx = random.randint(0,len(img_paths))

    img = cv2.imread(img_paths[rand_idx])

    img = cv2.resize(img, (args.img_width,args.img_width))

    img = img/255.

    csv_data_path = os.path.join(args.base_data_dir,args.input_data_dir_csv + "*")

    csv_paths = sorted(glob.glob(csv_data_path),key=natural_keys)

    X_img = np.empty((1, args.img_width, args.img_width, args.channels))
    X_csv = np.empty((1, 52,3))
    X_csv_weight = np.empty((1,52,3))

    X_img[0,] = img

    with open(csv_paths[rand_idx]) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j,row in enumerate(csv_reader):
                    for k, val in enumerate(row[4:7]):
                      X_csv[0,j,k] = float(val)/360.
                      X_csv_weight[0,j,k] = float(row[7])





    return ([X_img, X_csv, X_csv_weight],), X_csv

def get_random_samples(args, number):

    img_data_path = os.path.join(args.base_data_dir,args.input_data_dir_img + "*")

    img_paths = sorted(glob.glob(img_data_path),key=natural_keys)

    random_indices = random.sample(range(0, len(img_paths)),number)

    csv_data_path = os.path.join(args.base_data_dir,args.input_data_dir_csv + "*")

    csv_paths = sorted(glob.glob(csv_data_path),key=natural_keys)

    X_img = np.empty((number, args.img_width, args.img_width, args.channels))
    X_csv = np.empty((number, args.csv_dims))

    output_samples = []

    for i, index in enumerate(random_indices):

        img = cv2.imread(img_paths[index])

        img = cv2.resize(img, (args.img_width,args.img_width))

        img = img/255.

        all_values = []

        with open(csv_paths[index]) as file:
                    csv_reader = csv.reader(file, delimiter=',')
                    for row in csv_reader:
                        if(len(row)>0):
                            values = row[1:]
                            for value in values[3:]:
                                all_values.append(float(value)/360.)

        X_img[i,] = img
        X_csv[i,]= all_values

    return ([X_img, X_csv],)



