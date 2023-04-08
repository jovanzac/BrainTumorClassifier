import cv2
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


LABEL_DIR = os.path.join("data", "brain_tumor.csv")
IMAGE_DIR = os.path.join("data", "BrainTumor")

def load_data() :
    # Read the dataset and extract all the labels
    data = pd.read_csv(LABEL_DIR)
    labels = data.Class

    # Extract all the image names from the directory
    scans = [f"Image{i+1}.jpg" for i in range(len(os.listdir(IMAGE_DIR)))]

    # Create a DataFrame of the images and their labels
    df = pd.DataFrame(scans)
    df.columns = ["Images"]
    df["Labels"] = labels
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def process_img(scans=None, labels=None) :
    print("in preprocess")
    nlabels, ndata = [], []
    deleted = []
    labels = list(labels.Labels)
    scans = scans.Images

    i = 0
    print(f"scans.shape: {scans.shape} ; labels: {len(labels)}")
    for i, img in enumerate(scans) :
        print(str(img))
        img = cv2.imread(os.path.join("data", "BrainTumor",str(img)))
        print(img.shape)
        img = cv2.resize(img, (240, 240))
        if img.shape[2] == 1 :
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/225
        
        print(f"i: {i}")
        try :
            print(labels[i])
            if labels[i] == 1 :
                label = to_categorical(1, num_classes=2)
            else :
                label = to_categorical(0, num_classes=2)
        except KeyError :
            print("Keyerror here")
            deleted.append([i, img])
            continue
        nlabels.append(label)
        ndata.append(img)

    ndata = np.array(ndata)
    nlabels = np.array(nlabels)
    
    print(deleted)
    
    return ndata, nlabels


def data_gen(ndata, nlabels, batch_size) :
    n = len(ndata)
    steps = n//batch_size
    indices = np.arange(n)
    
    i = 0
    while True :
        # Get the next batch
        next_batch = indices[i*batch_size: (i+1)*batch_size]
        xbatched = ndata[next_batch[0]: next_batch[-1]]
        ybatched = nlabels[next_batch[0]: next_batch[-1]]
        
        i += 1
        yield xbatched, ybatched
        
        if i >=steps :
            i = 0