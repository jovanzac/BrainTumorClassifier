import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
import data_preprocessing as dp


EPOCHS = 10
IMG_WIDTH = 240
IMG_HEIGHT = 240
NUM_CATEGORIES = 2
TRAIN_SIZE = 0.7
BATCH_SIZE = 32


def cnn_model() :
    model = tf.keras.models.Sequential(
        [
            # First Convolutional layer and max pooling
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # Second Convolutional layer and max pooling
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            # Flatenning
            tf.keras.layers.Flatten(),
            # Hidden layers
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            # Dropout
            tf.keras.layers.Dropout(0.5),
            # Output Layer
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ]
    )
    # Compile neural network
    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )
    
    return model


def load_and_preprocess(set_type) :
    # Load and preprocess data
    data_frame = dp.load_data()
    # Create the training and testing sets
    indices = math.ceil(TRAIN_SIZE * data_frame.shape[0])
    if set_type :
        x_train = data_frame.iloc[:indices, :1]
        y_train = data_frame.iloc[:indices, 1:]
        x_train, y_train = dp.process_img(scans=x_train, labels=y_train)
        with open(os.path.join("data", "train_sets.pkl"), "wb") as pf :
            pickle.dump((x_train, y_train), pf)
        
        return x_train, y_train
        
    else :
        x_test = data_frame.iloc[indices:, :1]
        y_test = data_frame.iloc[indices:, 1:]
        x_test, y_test = dp.process_img(scans=x_test, labels=y_test)
        with open(os.path.join("data", "test_sets.pkl"), "wb") as pf :
            pickle.dump((x_test, y_test), pf)
            
        return x_test, y_test


def main() :
    try :
        with open(os.path.join("data", "train_sets.pkl"), "rb") as pf :
            x_train, y_train = pickle.load(pf)
    except Exception :
        print("Images already preprocessed")
        x_train, y_train = load_and_preprocess(set_type=1)
    try :
        with open(os.path.join("data", "test_sets.pkl"), "rb") as pf :
            x_test, y_test = pickle.load(pf)
    except Exception :
        print("Images already preprocessed")
        x_test, y_test = load_and_preprocess(set_type=0)
    
    # Shuffle the training and testing datasets
    map(lambda x: np.random.shuffle(x), [x_train, y_train, x_test, y_test])
    
    # Find the number of training steps
    nb_train_steps = x_train.shape[0]//BATCH_SIZE
    # Initialize the model
    model = cnn_model()
    # Fit model on the training data
    model.fit_generator(
        dp.data_gen(ndata=x_train, nlabels=y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        steps_per_epoch=nb_train_steps
    )
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)
    
    # Save model
    model.save(os.path.join("data", "trained_model"))


if __name__ == "__main__" :
    main()