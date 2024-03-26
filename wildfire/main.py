from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import os
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras 
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import my


app = Flask(__name__)


def create_df_img(filepath):
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepath))
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    df = pd.concat([filepath, labels], axis=1)
    return df


@app.route('/')
def html_file():
    return render_template('Main.html')


@app.route('/process_image', methods=['POST'])
def process_image():

    image_upload = request.files['image']
    file_path = 'Imagefile/' + image_upload.filename

    # Create a directory if it doesn't exist
    if not os.path.exists('Imagefile/'):
        os.makedirs('Imagefile/')

    image_upload.save(file_path)
  
    # Make predictions using the Wildfire Model
    predictions = my.predictImage(file_path)

    os.remove(file_path)

    # Process the predictions
    if predictions > 0.80:
        result = f'Wildfire (Probability: {predictions})'
    else :
        result = f'No Wildfire (Probability: {predictions})'

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
