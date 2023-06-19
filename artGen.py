import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def stylize_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load pre-trained model
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    files = list(os.listdir(input_dir))
    # Iterate over the files in the input directory
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        content_image = plt.imread(input_path)
        style_image = plt.imread(input_path)

        content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
        style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

        style_image = tf.image.resize(style_image, (256, 256))

        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        # Convert the tensor to a NumPy array
        stylized_image = tf.squeeze(stylized_image, axis=0).numpy()

        # Postprocess the stylized image
        stylized_image = (stylized_image) * 255.0
        stylized_image = stylized_image.astype('uint8')

        # Save the stylized image
        output_image = Image.fromarray(stylized_image)
        output_image.save(output_path)
        print(f"{i}/{len(files)} Gerado arquivo {output_path}")


# Specify the input and output directories
input_directory = 'images/Pablo_Picasso'
output_directory = 'output'

# Call the function to stylize the images
stylize_images(input_directory, output_directory)

