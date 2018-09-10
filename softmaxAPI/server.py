import os

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import request, Flask, jsonify, render_template

import model

UPLOAD_FOLDER = '/home/minhto/softmaxAPI/data/test_images'
x = tf.placeholder("float", [None, 784])
sess = tf.Session()


# restore
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "data/regression.ckpt")

def regression(input):
    return sess.run(y1, feed_dict={x: input})

# webapp
app = Flask(__name__)

@app.route('/upload')
def upload_file1():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        img = Image.open(f)
        input = ((255 - np.array(img)) / 255.0).reshape(1, 784)
        output = regression(input)
        prediction = np.argmax(output)
        prediction_ = prediction.tolist()
    return jsonify(prediction_)

if __name__ == '__main__':
    app.run(debug=True)
