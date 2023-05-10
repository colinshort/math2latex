import os
import pickle
import cv2
import numpy as np

# cnn_model = pickle.load(open('../models/cnn_model.pkl', 'rb'))
class Component:
    def __init__(self, component, centroid):
        self.component = component
        self.centroid = centroid

path = '../inputs/'
for i, img in enumerate(os.listdir(path)):
    image = cv2.imread((os.path.join(path, img)))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out_shape = img.shape
    img = cv2.GaussianBlur(img, (3, 3), 0)
    threshold = cv2.threshold(img, 0, 244, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)

    component_list = []

    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        if (area > 50) and (area < 15000):
            component = np.zeros(out_shape, dtype="uint8")
            mask = (label_ids == i).astype("uint8")*255
            component = cv2.bitwise_or(component, mask)
            component = cv2.bitwise_not(component, mask)
            component_obj = Component(component, centroid[i,])
            component_list.append(component_obj)       

    i = 0
    while i < len(component_list) - 1:
        curr = component_list[i]
        currX = curr.centroid[0]
        currY = curr.centroid[1]
        j = i + 1
        while j < len(component_list):
            next = component_list[j]
            nextX = next.centroid[0]
            nextY = next.centroid[1]
            if abs(currX - nextX) < 60 and abs(currY - nextY) < 100:
                curr.component = cv2.bitwise_and(curr.component, next.component)
                component_list.remove(next)
            else:
                j += 1
        i+=1

    for c in component_list:
        cv2.imshow("Final", c.component)
        cv2.waitKey(0)