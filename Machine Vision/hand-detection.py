import numpy as np
import cv2 as cv
import os
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

drawing = False # true if mouse is pressed
clasa = True # if True, chose dashed line claas. Press 'c' to toggle
undo = True
color = (0,255,0)
ix,iy = -1,-1
rectangles = {'class':[],
                'xmin':[],
                'ymin':[],
                'xmax':[],
                'ymax':[]}
# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,rectangles
    global unduimg,img, undo
    if event == cv.EVENT_LBUTTONDOWN:
        unduimg = deepcopy(img)
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            img = deepcopy(unduimg)
            cv.rectangle(img,(ix,iy),(x,y),color)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        undo = True
        cv.rectangle(img,(ix,iy),(x,y),color)
        line_type = 'black' if clasa else 'me'
        xmin,xmax = (ix,x) if ix < x else (x,ix)
        ymin,ymax = (iy,y) if iy < y else (y,iy)
        values = [line_type,xmin,ymin,xmax,ymax]
        for k, v in zip(rectangles, values):
            rectangles[k].append(v)

def safe_rectangles(im_name):
    global rectangles
    df = pd.DataFrame(rectangles)
    df.to_csv(im_path+'/csv/'+im_name[:-4]+'.csv',index=False)
    for key in rectangles: 
        rectangles[key].clear()


im_path = r'C:\Users\vase_\Downloads\jetbot-nano\Data\vid2'
index = 0
im_name = str(index) + '.jpg'
img = unduimg = cv.imread(os.path.join(im_path, im_name))
cv.namedWindow('image')
cv.setMouseCallback('image',draw_rectangle)

while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('c'):
        clasa = not clasa
        color = (0,255,0) if clasa else (0,0,255)
    if k == ord('z'):
        safe_rectangles(im_name)
        index+=1
        im_name = str(index) + '.jpg'
        img = unduimg = cv.imread(os.path.join(im_path, im_name))
    if k == ord('x'):
        if undo:
            img = deepcopy(unduimg)
            for key in rectangles: 
                rectangles[key].pop()
            undo = False
    elif k == 27:
        break
cv.destroyAllWindows()