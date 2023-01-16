###to run the Flask server
# FLASK_APP=app.py flask run
# https://sentimentanalyseapi.herokuapp.com/api?my_tweet=I+hate+you
# http://127.0.0.1:5000/api?my_tweet=I+hate+you
###
import io
import numpy as np
import zlib
import albumentations as aug
import segmentation_models as sm
import my_classes as mc
from flask import Flask, request, Response
import os

app = Flask(__name__)

# ## HELPERS

def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        aug.PadIfNeeded(384, 480)
    ]
    return aug.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        aug.Lambda(image=preprocessing_fn),
    ]
    return aug.Compose(_transform)


@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('loading model and data')
    # load best wehts
    BACKBONE = 'efficientnetb3'
    preprocess_input = sm.get_preprocessing(BACKBONE)


    global model, dataset
    model = sm.Unet(BACKBONE, classes=8, activation='softmax')
    model.load_weights('best_model.h5')

    x_test_dir = 'images.txt'
    y_test_dir = 'mask.txt'

    dataset = mc.Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )



def predict(image_id):
    # Prediction:
    image, mask = dataset[image_id]

    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image)

    return np.argmax(pr_mask.squeeze(), 2)

# API
@app.route("/api")
def auto_car():

    img_id = int(request.args.get("image_id"))
    if not img_id:
        img_id = 0

    pr_mask = predict(img_id)
    resp, _, _ = compress_nparr(pr_mask)

    return Response(response=resp, status=200,
                    mimetype="application/octet_stream")




if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
