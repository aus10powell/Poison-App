from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import io, os
import logging
import keras_metrics
from tensorflow import keras
import utils
## Configs
keras.utils.get_custom_objects()['recall'] = utils.recall
keras.utils.get_custom_objects()['precision'] = utils.precision
keras.utils.get_custom_objects()['f1'] = utils.f1


def teachable_machine_classification(img=None, model=None):
    """Performs inference on image uploaded"""

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 299, 299, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (299, 299)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    print("Prediction", prediction)
    return prediction[0][
        1
    ]  # np.argmax(prediction) # return position of the highest probability


def load_model(weights_file=None):
    """Loads trained keras model"""
    dependencies = {
        "binary_f1_score": keras_metrics.binary_f1_score,
        "binary_precision": keras_metrics.binary_precision,
        "binary_recall": keras_metrics.binary_recall,
    }

    try:
        assert os.path.exists(weights_file), f"File '{weights_file}' does not exist"
        # Load the model
        model = keras.models.load_model(
            weights_file, custom_objects=dependencies, compile=False
        )

        return model
    except Exception as e:
        logging.error("ERROR: ", e)
        print("ERROR: ", e, " Failed to load ML model")

        return None
