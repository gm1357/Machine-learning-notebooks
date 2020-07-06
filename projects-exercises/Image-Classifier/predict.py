import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import json
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Application to predict flower images using TensorFlow',
    )

    parser.add_argument('path_to_image', action='store',
                        help='Path to the image to be predicted')

    parser.add_argument('model', action='store',
                        help='Path to the model to be used as predictor')

    parser.add_argument('--top_k', action='store',
                        dest='top_k',
                        default=1,
                        type=int,
                        help='Number of most likely classes to show (default: 1)')

    parser.add_argument('--category_names', action='store',
                        dest='labels_path',
                        help='Path to a JSON file mapping labels to flower names')

    return parser.parse_args()

def process_image(image):
    imageTensor = tf.convert_to_tensor(image)
    imageTensor = tf.image.resize(imageTensor, (224, 224))
    imageTensor /= 255
    
    return imageTensor.numpy()

def predict(image_path, path_to_model, top_k):
    try:
        im = Image.open(image_path)
    except:
        print('Could not open image at path: ', image_path)
        exit()
        
    im = process_image(np.asarray(im))
    im = tf.expand_dims(im, 0)
    
    try:
        loaded_model = tf.keras.models.load_model(path_to_model, custom_objects={'KerasLayer':hub.KerasLayer})
    except:
        print('Could not open model at path: ', path_to_model)
        exit()
    
    predictions = loaded_model.predict(im)
    values, indices = tf.nn.top_k(predictions, k=top_k)
    
    return values.numpy()[0], (indices+1).numpy().astype(str)[0]

def load_class_names(path):
    class_names = None
    if (path):
        with open('label_map.json', 'r') as f:
            class_names = json.load(f)
    return class_names

def main():
    arguments = parse_arguments()
    
    class_names = load_class_names(arguments.labels_path)

    probs, classes = predict(
        arguments.path_to_image,
        arguments.model,
        arguments.top_k
    )
    
    if (class_names):
        classes = np.array([class_names[class_val] for class_val in classes])

    for index in range(probs.size):
        print(classes[index], ': {:.3%}'.format(probs[index]))

if __name__ == "__main__":
    main()