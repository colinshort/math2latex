import os
import numpy as np
import cv2
import pandas as pd

path = '/Users/col_s/CSCI_575_Project_Data/archive/extracted_images/'
data = os.listdir(path)
features = []
labels = []
n = 0
for folder in data:
    currentFolder = path + folder
    for i, file in enumerate(os.listdir(currentFolder)):
        im = cv2.imread((os.path.join(currentFolder, file)))
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        image = np.array(img)
        image = image.ravel().tolist()
        features.append(image)
        labels.append(n)
    n+=1
df = pd.DataFrame(features)
df["label"] = labels

df = df.sample(frac=1)
df.to_pickle('../data/data.pkl')