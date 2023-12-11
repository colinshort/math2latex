import os
import sys
import pickle
import csv
import cv2
import numpy as np

verbose = False
if len(sys.argv) == 2:
    if sys.argv[1] == "-v" or sys.argv[1] == "--verbose":
        verbose = True

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
    if verbose:
        cv2.imwrite("../outputs/grayscale-img" + str(i) + ".png", img)
    out_shape = img.shape
    img = cv2.GaussianBlur(img, (3, 3), 10)
    if verbose:
        cv2.imwrite("../outputs/blurred-img" + str(i) + ".png", img)
    threshold = cv2.threshold(img, 0, 244, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if verbose:
        cv2.imwrite("../outputs/binary-img"  + str(i) + ".png", img)
    kernel = np.ones((3, 3))
    proc_img = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    if verbose:
        cv2.imwrite("../outputs/closing-img"  + str(i) + ".png", proc_img)
    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(proc_img, 4, cv2.CV_32S)

    component_list = []

    for j in range(1, totalLabels):
        area = values[j, cv2.CC_STAT_AREA]
        x1 = values[j, cv2.CC_STAT_LEFT]
        y1 = values[j, cv2.CC_STAT_TOP]
        w = values[j, cv2.CC_STAT_WIDTH]
        h = values[j, cv2.CC_STAT_HEIGHT]
        if (area > 50) and (area < 15000):
            component = np.zeros(out_shape, dtype="uint8")
            mask = (label_ids == j).astype("uint8")*255
            component = cv2.bitwise_or(component, mask)
            component = cv2.bitwise_not(component, mask)
            component_obj = Component(component, centroid[j,], x1, y1, w, h)
            component_list.append(component_obj)

    j = 0
    while j < len(component_list) - 1:
        curr = component_list[j]
        currX = curr.centroid[0]
        currY = curr.centroid[1]
        k = j + 1
        while k < len(component_list):
            next = component_list[k]
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
                k += 1
        j+=1
    
    for c in component_list:
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

    for j in range(0, len(component_list) - 1):
        for k in range(j, len(component_list)):
            if component_list[k].left < component_list[j].left:
                temp = component_list[j]
                component_list[j] = component_list[k]
                component_list[k] = temp

    component_imgs = np.zeros(shape=(len(component_list), 45, 45))
    j = 0
    for c in component_list:
        curr_component = (c.component)
        curr_component = cv2.copyMakeBorder(curr_component, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        curr_component = cv2.GaussianBlur(curr_component, (5, 5), 10)

        _, curr_component = cv2.threshold(curr_component, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        curr_component = cv2.ximgproc.thinning(curr_component)
    
        curr_component = curr_component.astype(np.uint8)
        curr_component = cv2.bitwise_not(curr_component)
    
        curr_component = curr_component[10:-10, 10:-10]

        component_imgs[j] = curr_component
        j += 1
    
    y_pred = cnn_model.predict(component_imgs.reshape(-1, 45, 45, 1))
    predictions = y_pred.argmax(axis=1)
    result = ""
    j = 0
    for pred in predictions:
        for pair in label_pairs:
            if int(pair[0]) == pred:
                result = result + pair[1] + " "
                if verbose:
                    cv2.imwrite("../outputs/component " + str(i) + "-" + str(j) +" - " + str(pair[1]) + ".png", component_imgs[j])
        j += 1
    results.append(result)

print(results)