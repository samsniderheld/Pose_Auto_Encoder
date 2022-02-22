import cv2
import numpy as np
import os
import csv
import pandas as pd
import json
from datetime import datetime
from tabulate import tabulate

def generate_test_image(model, test_input, path, epoch):

  sample = np.expand_dims(test_input,0)

  prediction = model(sample)
    
  predicted_image = np.uint8(prediction[0] * 255.)

  predicted_image = cv2.resize(predicted_image,(256,144)) 

  test_input = cv2.resize(test_input,(256,144)) * 255.


  output_image = np.concatenate((test_input, predicted_image), axis=1)

  output_path = os.path.join(path,f"epoch_{epoch:04d}_image.jpg")

  cv2.imwrite(output_path,output_image)


def generate_test_image_grid(model, test_input, path, epoch):

  sample = np.array(test_input)

  prediction = model(sample)
    
  predicted_samples = np.uint8(prediction * 255.)

  rows = []
  for i in range(0, 100, 10):
    rows.append(np.concatenate(predicted_samples[i:i+10], axis = 1))

  concat_image = np.concatenate(rows, axis = 0)


  output_path = os.path.join(path,f"epoch_{epoch:04d}_image_grid.jpg")

  cv2.imwrite(output_path,concat_image)


def generate_dual_test_image(model, test_input, path, epoch):
  
  prediction = model(test_input)
    
  predicted_image = np.uint8(prediction[0][0] * 255.)

  # predicted_image = cv2.resize(predicted_image,(32,32)) 

  test_img = np.uint8(test_input[0][0][0]) * 255.

  output_image = np.concatenate((test_img, predicted_image), axis=1)

  output_image = cv2.resize(output_image,(512,256))

  output_img_path = os.path.join(path,f"epoch_{epoch:04d}_image.jpg")

  cv2.imwrite(output_img_path,output_image)

  csv_dif = np.absolute(test_input[0][1]-prediction[1])

  csv_mean = np.mean(csv_dif)

  return csv_mean

def generate_dual_test_image_grid(model, test_input, path, epoch):

  prediction = model(test_input)

  predicted_samples = np.uint8(prediction[0] * 255.)

  rows = []
  for i in range(0, 100, 10):
      rows.append(np.concatenate(predicted_samples[i:i+10], axis = 1))

  concat_image = np.concatenate(rows, axis = 0)


  output_path = os.path.join(path,f"epoch_{epoch:04d}_image_grid.jpg")

  cv2.imwrite(output_path,concat_image)

def generate_bone_accuracy_table(model, test_input, csv_values, path, epoch, print_to_terminal,just_bones):

  prediction = model(test_input)

  if just_bones:
    outputs = prediction[0].numpy().flatten()
  else:
    outputs = prediction[1][0]

  dict = { 
    'Input Values': csv_values.flatten(),
    'Predictions': outputs
  }

  df = pd.DataFrame(dict)

  output_csv_path = os.path.join(path,f"epoch_{epoch:04d}_bones.csv")

  df.to_csv(output_csv_path)  

  if print_to_terminal:
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

def save_experiment_history(args, history, path):

  experiment = {

      'notes': args.notes,
      'latent_dimensions': args.latent_dim,
      'number_of_epochs': args.num_epochs,
      'batch_size': args.batch_size,
      'img_width': args.img_width,
      'csv_dims': args.csv_dims,
      'loss_history': history

  }

  file_name = args.experiment_name + ".json"

  output_path = os.path.join(path,file_name)

  with open(output_path, 'w') as outfile:
    json.dump(experiment, outfile)







