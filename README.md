# Cats and Dogs Image Classification with CNN

This project builds a Convolutional Neural Network (CNN) model to classify images of cats and dogs using TensorFlow and Keras. The CNN model is trained on a dataset of labeled cat and dog images, and then it is used to predict whether new images depict a cat or a dog.

## Dataset

The dataset used for training and testing is a set of labeled images of cats and dogs. It should be organized in directories for easy loading:
- `train/`: Contains subdirectories for `cats/` and `dogs/`, each with respective images for training.
- `test/`: Contains subdirectories for `cats/` and `dogs/`, each with respective images for testing.

### Dataset structure:

```plaintext
/train
  /cats
    cat_1.jpg
    cat_2.jpg
    ...
  /dogs
    dog_1.jpg
    dog_2.jpg
    ...
/test
  /cats
    cat_3.jpg
    ...
  /dogs
    dog_4.jpg
    ...
```

## Approach
1. Data Augmentation: Both the training and testing images are preprocessed and augmented using the ImageDataGenerator class from Keras. Techniques like rescaling, horizontal flipping, shearing, and zooming are applied to make the model more robust.

2. Building the CNN:

- Convolutional Layers: The model starts with two convolutional layers, each followed by a max-pooling layer, which helps extract features from the images.
- Flattening: The 2D feature maps are converted into 1D vectors before passing them into a fully connected layer.
- Fully Connected Layers: The model has one hidden dense layer with 128 units and a ReLU activation function.
- Output Layer: The final layer contains a single unit with a sigmoid activation function to classify the images as either a cat or a dog.
  
3. Training: The model is compiled with the adam optimizer and trained using binary cross-entropy loss. The training process runs for 25 epochs.

4. Prediction: A sample image can be loaded and passed into the trained model to predict whether it's a cat or a dog.

## Requirements
Install the necessary libraries using pip:

```bash
pip install -r requirements.txt
```

## Instructions
1. Prepare the Dataset: Ensure your dataset is organized in the specified structure and located in the appropriate paths.
2. Run the Script: Execute the Python script to train the model on the dataset.
Make Predictions: After training, the model can be used to predict whether a given image is of a cat or a dog.

### Example:
```python
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load and preprocess the image
test_image = load_img('/path/to/image.jpg', target_size=(64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = cnn.predict(test_image)
if result[0][0] == 1:
    print('Dog')
else:
    print('Cat')
```










