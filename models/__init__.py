from tensorflow.keras.models import load_model
import tensorflow.config as tfconfig
import base64
from io import BytesIO
from PIL import Image
from io import BytesIO
import numpy as np

IMG_SIZE = (128, 128)

# needs updation
labels = {
    0: 'Pepper__bell___Bacterial_spot', 
    1: 'Pepper__bell___healthy', 
    2: 'Potato___Early_blight', 
    3: 'Potato___healthy', 
    4: 'Potato___Late_blight', 
    5: 'Tomato_Bacterial_spot', 
    6: 'Tomato_Early_blight', 
    7: 'Tomato_healthy', 
    8: 'Tomato_Late_blight', 
    9: 'Tomato_Leaf_Mold', 
    10: 'Tomato_Septoria_leaf_spot', 
    11: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    12: 'Tomato__Target_Spot', 
    13: 'Tomato__Tomato_mosaic_virus', 
    14: 'Tomato__Tomato_YellowLeaf__Curl_Virus'
    }


def get_model(path):
    return load_model(path)

def set_memory_limit(gigs=1):
    MEMORY_LIMIT = 1024*gigs
    try:
        gpus = tfconfig.list_physical_devices("GPU")
        if gpus:
            tfconfig.experimental.set_virtual_device_configuration(
                gpus[0],
                [tfconfig.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)]
            )
            print(f"GPU is set to {MEMORY_LIMIT/1024}GB")
    except RuntimeError as e:
        print(e)

def data_uri_to_image(uri):
    if uri.startswith("data:image"):
        uri = uri.split(",")[1]
    image_data = base64.b64decode(uri)
    image = Image.open(BytesIO(image_data))
    return image

def preprocess(image):
    # if not isinstance(image, Image.Image):
    #     raise ValueError("Input to preprocess must be a PIL.Image object")
    
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def get_class(predictions):
    return labels[np.argmax(predictions)]
