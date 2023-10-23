import os
import pickle
import csv
import cv2
import numpy as np

cnn_model = pickle.load(open('../models/cnn_model.pkl', 'rb'))
label_pairs = []
with open('../data/labels.csv', 'r') as csvfile:
    csvread = csv.reader(csvfile)
    for line in csvfile:
        if len(line) > 1:
            line_arr = line.split(',')
            n = line_arr[0]
            folder = line_arr[1]
            folder = folder.replace('\n', '')
            if len(folder) > 1:
                folder = "\\" + folder
            if folder == '\\ascii_124':
                folder = '\\|'
            label_pairs.append([n, folder])
csvfile.close()

class Component:
    def __init__(self, component, centroid, left, top ,width ,height):
        self.component = component
        self.centroid = centroid
        self.left = left
        self.top = top
        self.width = width
        self.height = height

path = '../inputs/'
results = []
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
        x1 = values[i, cv2.CC_STAT_LEFT]
        y1 = values[i, cv2.CC_STAT_TOP]
        w = values[i, cv2.CC_STAT_WIDTH]
        h = values[i, cv2.CC_STAT_HEIGHT]
        if (area > 50) and (area < 15000):
            component = np.zeros(out_shape, dtype="uint8")
            mask = (label_ids == i).astype("uint8")*255
            component = cv2.bitwise_or(component, mask)
            component = cv2.bitwise_not(component, mask)
            component_obj = Component(component, centroid[i,], x1, y1, w, h)
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
                curr.centroid = abs(curr.centroid + next.centroid)/2
                curr.width = int(max(curr.left + curr.width, next.left + next.width) - min(curr.left, next.left))
                if (curr.top + curr.height) <= (next.top + next.height):
                    curr.height = int(curr.height + next.height + abs((curr.top + curr.height) - next.top))
                else:
                    curr.height = int(curr.height + next.height + abs((next.top + next.height) - curr.top))
                curr.left = min(curr.left, next.left)
                curr.top = min(curr.top, next.top)
                component_list.remove(next)
            else:
                j += 1
        i+=1

    
    for c in component_list:
        new_img = image.copy()
        pt1 = (c.left, c.top)
        c.component = c.component[c.top:c.top+c.height, c.left:c.left+c.width]
        if c.width > c.height:
            max_dim = c.width
            if max_dim < 45:
                max_dim = 45
            pad = max_dim - c.height
            c.component = cv2.copyMakeBorder(c.component, int(pad/2), int(pad/2), 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        else:
            max_dim = c.height
            if max_dim < 45:
                max_dim = 45
            pad = max_dim - c.width
            c.component = cv2.copyMakeBorder(c.component, 0, 0, int(pad/2), int(pad/2), cv2.BORDER_CONSTANT, value=(255, 255, 255))
        c.component = cv2.resize(c.component, (45, 45))

    for i in range(0, len(component_list) - 1):
        for j in range(i, len(component_list)):
            if component_list[j].left < component_list[i].left:
                temp = component_list[i]
                component_list[i] = component_list[j]
                component_list[j] = temp

    component_imgs = np.zeros(shape=(len(component_list), 45, 45))
    i = 0
    for c in component_list:
        component_imgs[i] = (c.component)
        i += 1
    
    # to view each symbol individually
    # for img in component_imgs:
    #     cv2.imshow("Component", img)
    #     cv2.waitKey(0)
    
    y_pred = cnn_model.predict(component_imgs.reshape(-1, 45, 45 ,1))
    predictions = y_pred.argmax(axis=1)
    result = ""
    for pred in predictions:
        for pair in label_pairs:
            if int(pair[0]) == pred:
                result = result + pair[1]
    results.append(result)
print(results)