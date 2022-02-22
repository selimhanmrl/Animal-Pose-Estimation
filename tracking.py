import os
from unittest import result 
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from draw_polygon import PolygonDrawer
import math
import pandas as pd
import sys
from demo import demo 
if len(sys.argv)>1:
    video_name = sys.argv[1]

if len(sys.argv)>2:
    exp_type = sys.argv[2]
if len(sys.argv)>3:
    actual_area = sys.argv[3]

if not os.path.exists('Results'):
    os.makedirs('Results')
if os.path.isfile("keypoints/keypoints_outputs_"+os.path.splitext(os.path.basename(video_name))[0]+".csv"):
    keypoint_file = "keypoints/keypoints_outputs_"+os.path.splitext(os.path.basename(video_name))[0]+".csv"
else:
    demo(video_name)
    keypoint_file = "keypoints/keypoints_outputs_"+os.path.splitext(os.path.basename(video_name))[0]+".csv"

cap = cv2.VideoCapture(video_name)
df = pd.read_csv(keypoint_file)

_, frame1 = cap.read()
pd_ = PolygonDrawer("Polygon",frame1)
image = pd_.run(frame1)

#fps for calculate time
fps = cap.get(5)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
pts = []
if exp_type == 'plusmaze':

    polyRight = Polygon(pd_.points[0])

    polyleft = Polygon(pd_.points[1])

    polyUp = Polygon(pd_.points[2])

    polyDown = Polygon(pd_.points[3])


if exp_type == 'openfield':
                  
    polyOutside  = Polygon(pd_.points[0])

    polyInside = Polygon(pd_.points[1])



total_path=0
right_= 0 
left_= 0
up_ = 0
down_ = 0
mid_ = 0

r_x = 0
l_x = 999
if exp_type == 'plusmaze':
    for i in pd_.points[0]:
        x,y = i 
        if x > r_x:
            r_x = x

    for i in pd_.points[1]:
        x,y = i 
        if x < l_x:
            l_x = x

if exp_type == 'openfield':
    for i in pd_.points[0]:
        x,y = i 
        if x > r_x:
            r_x = x
        if x < l_x:
            l_x = x
    
        
flag = 0
inside = 0 
outside = 0
dist_exp_area = abs(r_x - l_x)
if exp_type == 'plusmaze':
    place = []
    for i in range(len(df)):
        p = Point(int(df["body_x"][i]),int(df["body_y"][i]))
        pts.append((int(df["body_x"][i]),int(df["body_y"][i])))

        if p.within(polyRight):
            right_ += 1
            if len(place) <1:
                place.append("Right Side")
        elif p.within(polyleft):
            left_ += 1
            if len(place) <1:
                place.append("Left Side")
        elif p.within(polyUp):
            up_ += 1
            if len(place) <1:
                place.append("Up Side")
        elif p.within(polyDown):
            down_ += 1
            if len(place) <1:
                place.append("Down Side")
        else:
            mid_ += 1
if exp_type == 'openfield':
    for i in range(len(df)):
        p = Point(int(df["body_x"][i]),int(df["body_y"][i]))
        pts.append((int(df["body_x"][i]),int(df["body_y"][i])))
    
        if p.within(polyInside):
            inside += 1
        else:
            outside += 1

for i in range(len(pts)-1):
    dist = math.sqrt((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)
    total_path+=dist

df_plus = pd.DataFrame(columns=["video_name","right_side_time","left_side_time","up_side_time","down_side_time","mid_time","total_time_in_opened_area","total_time_in_closed_area","first_enterence","total_distance"])
df_open = pd.DataFrame(columns=["video_name","Inside_time","outside_time","total_distance"])
if os.path.isfile("Results/plusmaze.csv"):
    df_plus = pd.read_csv("Results/plusmaze.csv")
if os.path.isfile("Results/openfield.csv"):
    df_open = pd.read_csv("Results/openfield.csv")

if exp_type == 'plusmaze':

    result = [  
                os.path.splitext(os.path.basename(video_name))[0],
                round(right_/fps,2),
                round(left_/fps,2),
                round(up_/fps,2),
                round(down_/fps,2),
                round(mid_/fps,2),
                round((left_+right_)/fps,2),
                round((up_+down_)/fps,2),
                place[0],
                round((total_path*int(actual_area))/dist_exp_area,4)]
    df_plus.loc[len(df_plus.index)] = result
    df_plus.to_csv('Results/plusmaze.csv',index=False)
    print("Results are done!!!")

if exp_type == 'openfield':
    result = [                                
                os.path.splitext(os.path.basename(video_name))[0],
                round(inside/fps,2),
                round(outside/fps,2),
                round((total_path*int(actual_area))/dist_exp_area,4)]   
    df_open.loc[len(df_open.index)] = result
    df_open.to_csv('Results/openfield.csv',index=False)
    print("Results are done!!!")


