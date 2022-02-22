# Keypoint-Prediction-Model
With own dataset create your own pose estimation model for any object which you labeled


First of all install requirements with 

    pip install -r requirements.txt

After installing needed libraries create own dataset

It depends on your object i will give example with mouses. 
In this example i labeled keypoints as     head, right ear, left ear, body, tail head, tail and hold all keypoints in csv file. 

Csv file's coloumns should be like '(Image-Path)', Keypoints' x locations, Keypoints' y locations

You should change some numbers in codes depends on your dataset. 

In model.py file at line 21 you should change number for what you want to predict in our example there is 6 keypoints and its 2 x and y coordinates so totally it is 12.

    self.l0 = nn.Linear(512,--)

After creating dataset save all images and coordinates in "labeled-data" folder 

We can start oour training part 

  in config.py folder you can change your epoch number as you want.
  Start Training with given code below
  
    python train.py
    
 After Creating Model we can analyze our videos 
 
(!!!! Our tracking code is specific for our owned model so it can give bad results for errors for make own model please contact me if you have problem.)
    Before analyzing you should change which part you will follow and change in code
    In tracking.py folder you should change part x and y coordinates in line 96-97 and 119-120
 
    python tracking.py 'video path' 'experiment type' 'experiment area exact lenght'
 
 For now as experiment are we use 'openfield' of 'plusmaze' while analyzing videos you should select areas for tracking.
 
 With mouse left click select point after changing are click mouse's left click for plus maze make it 4 time and for openfield make it 2 times 
 For plus maze experiment area select areas ordered as Left Right Up Down
 
  For open field experiment area select areas ordered as Outer area and inner area
 
 Example:

    python tracking.py OFT_12.mp4 openfield 100
    

You can check folder for keypoints result of experiments and video for pose estimation.

Demo video for pose estimation model in mouses

https://www.youtube.com/watch?v=M6zKXrCf1Xc


For asking any questions please contact me 
    
    selimhanmrl@gmail.com
