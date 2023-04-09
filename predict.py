import os
import cv2
import numpy as np
import tensorflow as tf


def process_img(img) :
    # os.path.join("data", "BrainTumor",str(img))
    print(img)
    try :
        img = cv2.imread(img)
        img = cv2.resize(img, (240, 240))
        if img.shape[2] == 1 :
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/225
        
        return np.array([img])
    
    except Exception as ex :
        print(ex)
    

def load_and_predict(img) :
    # Load the saved model
    model = tf.keras.models.load_model(os.path.join("data", "trained_model"))
    
    # Preprocess the image
    data = process_img(img)
    
    # Predict output based on the trained model parameters
    y_pred = model.predict(data)
    y_pred = np.where(y_pred > 0.5, 1,0)
    
    return "Malignant" if y_pred[0][0] == 0 else "Benign"


if __name__ == "__main__" :
    load_and_predict("Image4.jpg")