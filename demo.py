import torch
import numpy as np
import cv2
import os
import albumentations
import config
from model import NNModel
import pandas as pd
from tqdm import tqdm
import sys

if len(sys.argv)>1:
    file_name = sys.argv[1]

def demo(filename):
    model = NNModel(pretrained=False, requires_grad=False).to(config.DEVICE)
    # load the model checkpoint
    checkpoint = torch.load('model.pth',  map_location=torch.device('cpu') )
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
  
    cap = cv2.VideoCapture(file_name)

    if not os.path.exists('video_outputs'):
       os.makedirs('video_outputs')
    if not os.path.exists('keypoints'):
        os.makedirs('keypoints')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # set up the save file path
    save_path = "video_outputs/vid_keypoint_detection_" + os.path.splitext(os.path.basename(filename))[0] +".mp4"
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
    keypoints_f = []
    df = pd.DataFrame(columns=["head_x","head_y","R_ear_x","R_ear_y","L_ear_x","L_ear_y","body_x","body_y","tail_head_x","tail_head_y","tail_x","tail_y"])
    line = 0 

  
    pbar = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),desc="Exract Keypoints and Create Video")
    while(cap.isOpened()):
        # capture each frame of the video

            ret, frame = cap.read()
            if ret == True:
            
                with torch.no_grad():
                    image = frame
                    image = cv2.resize(image, (224, 224))
                    orig_frame = image.copy()
                    orig_h, orig_w, c = orig_frame.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image / 255.0
                    image = np.transpose(image, (2, 0, 1))
                    image = torch.tensor(image, dtype=torch.float)
                    image = image.unsqueeze(0).to(config.DEVICE)
                    outputs = model(image)
                outputs = outputs.cpu().detach().numpy()
                outputs = outputs.reshape(-1, 2)
                keypoints = outputs
                for p in range(keypoints.shape[0]):
                    cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                            1, (0, 0, 255), -1, cv2.LINE_AA)
                    x,y = keypoints[p, 0],keypoints[p, 1]
                    keypoints_f.append(x)
                    keypoints_f.append(y)
                orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))
                df.loc[line] = keypoints_f
                keypoints_f = []
                out.write(orig_frame)
                line += 1 
                # press `q` to exit
                if cv2.waitKey(27) & 0xFF == ord('q'):
                    break
                pbar.update(1)
            else: 
                break
    df.to_csv("keypoints/keypoints_outputs_"+ os.path.splitext(os.path.basename(filename))[0] + ".csv")
    cap.release()
    
    # close all frames and video windows
    cv2.destroyAllWindows()