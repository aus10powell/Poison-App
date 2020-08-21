import keras
from PIL import Image, ImageOps
import numpy as np
import io
import logging
import keras_metrics


def teachable_machine_classification(img, weights_file):
    dependencies = {'binary_f1_score': keras_metrics.binary_f1_score,'binary_precision': keras_metrics.binary_precision,'binary_recall': keras_metrics.binary_recall}
    # Load the model
    model = keras.models.load_model(weights_file, custom_objects=dependencies,compile=False)
    try:
        dependencies = {'binary_f1_score': keras_metrics.binary_f1_score,'binary_precision': keras_metrics.binary_precision,'binary_recall': keras_metrics.binary_recall}
        # Load the model
        model = keras.models.load_model(model_path, custom_objects=dependencies,compile=False)
        logging.info('Model successfully loaded')
    except:
        logging.warning('Model Failed to load')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print('Prediction',prediction)
    return prediction[0][1]#np.argmax(prediction) # return position of the highest probability
