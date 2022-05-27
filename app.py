# Libraries
import numpy as np
import matplotlib.pyplot as plt
import os, random
from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from seg_functions import infer

# Set parameters
class_name = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
img_height = 256
img_width = 512
n_classes = 8
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

# Flask app initialization
app = Flask(__name__)


@app.route('/')
def index():
    file_list = os.listdir('./static/images')
    return render_template('index.html',file_list=file_list)


@app.route('/prediction', methods=['POST'])
def predict():
    # Return selected file
    file = request.form['file']
    image_path = str('./static/images/'+ file)
    mask_path = str('./static/masks/'+ file)

    # Shared custom object of model
    def dice_coeff(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    # Mobilenet custom object
    def relu6(x):
        return K.relu(x, max_value=6)

    model = tf.keras.models.load_model(
        'model/mobilenet_segnet.h5', custom_objects={'dice_coeff':dice_coeff, 'accuracy':'accuracy', 'relu6':relu6})

    # Load and process mask
    mask = img_to_array(load_img(mask_path, target_size=(img_height, img_width), color_mode="grayscale"))
    mask = np.squeeze(mask)
    plt.imsave('./static/outputs/colorized_mask.png', mask, cmap='nipy_spectral_r')

    # Predict from image
    seg_img = infer(
        model=model, inp=image_path, out_fname='./static/outputs/prediction.png',
        n_classes=n_classes, colors=class_colors,
        prediction_width=512, prediction_height=256,
        read_image_type=1)
    plt.imsave('./static/outputs/prediction.png', seg_img, cmap='nipy_spectral_r')
    
    return render_template('prediction.html', image_data=file)

# Run app
if __name__ == '__main__':
    app.run(debug=True)