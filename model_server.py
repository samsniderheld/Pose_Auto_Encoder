from flask import Flask, current_app, request, send_file, Response
import json
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from scipy.spatial import distance
import scipy.misc
from keras.preprocessing import image
from Model.bone_variational_auto_encoder import create_variational_bone_auto_encoder
from Model.bone_auto_encoder import create_bone_auto_encoder

# encoder_model = tf.keras.models.load_model('Saved_Models/0300_encoder_model.h5')
# bone_decoder_model = tf.keras.models.load_model('Saved_Models/0300_bone_decoder_model.h5')

img_dim = 128

encoder, bone_decoder, auto_encoder = create_variational_bone_auto_encoder(
        dims=img_dim, latent_dim = 128)



# encoder, bone_decoder, auto_encoder = create_bone_auto_encoder(
#         dims=img_dim , latent_dim = 128)

auto_encoder.load_weights('Saved_Models/bone_auto_encoder_model.h5')


app = Flask(__name__)
@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.form['img']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400


    b64_decoded_img = base64.b64decode(data)

    byte_img = io.BytesIO(b64_decoded_img)

    pil_img= Image.open(byte_img)

    cv2.imwrite('test.jpg',np.array(pil_img))

    pil_img = pil_img.resize((img_dim,img_dim))

    np_img = image.img_to_array(pil_img)
    
    np_img = np_img/255.

    sample = np.expand_dims(np_img, axis=0)

    empty_CSV = np.empty((1,52,3))
   
    # prediction = bone_decoder_model(encoder_model(sample))
    prediction = auto_encoder([sample,empty_CSV,empty_CSV])

    response = {"bones": prediction[0].numpy().flatten().tolist()}

    return json.dumps(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)