import re
import pandas as pd
import numpy as np

path = './data/sample_coco.txt'

classes = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

lines = []
with open(path) as f:
    lines = f.readlines()

class my_dictionary(dict):
    def __init__(self):
        self = dict()
        
    def add(self, key, value):
        self[key] = value

def datarestructure(columns,dataList):
    ls = []
    dataDict = my_dictionary()
    for i in dataList:
        a = i.replace(',\n','')
        val = re.findall('[0-9]+', a)
        for cnt,dat in zip(columns,val):
            dataDict.add(cnt,dat)
        ls.append(dataDict.copy())
    return ls

def get_class(i):
    return classes[i]



def iou(x, centroids):
    dists = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            dist = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            dist = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            dist = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            dist = (c_w * c_h) / (w * h)
        dists.append(dist)
    return np.array(dists)

def avg_iou(ious):
    n = ious.shape[0]
    sums = 0.
    for i in range(n):
        # note IOU() will return array which contains IoU for each centroid and X[i]
        # slightly ineffective, but I am too lazy
        sums += max(ious[i])
    return sums / n



columns = ['id', 'height', 'width', 'x', 'y', 'bbox_width', 'bbox_height']
data = pd.DataFrame(datarestructure(columns,dataList=lines))

data.to_csv('./data/coco.csv',index=False)