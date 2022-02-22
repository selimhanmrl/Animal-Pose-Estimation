import numpy as np
import cv2

# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name,frame,count = 4):
        self.window_name = window_name # Name for our window
        self.frame = frame
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.polycount = count
        self.currentpoint = []


    ###
    #  Here with left button it select point for edge of polygon
    #  right button for finishing selection polygon and can draw new one
    #  with escape button finish selection
    ###
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        count = 0
        height, width, _ = self.frame.shape
        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 5, (0, 200, 0), -1)
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.currentpoint.append(((int(x*224)/width), int((y*224)/height)))
            print(self.currentpoint)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.points.append(self.currentpoint)
            self.currentpoint = []
            if self.polycount == count :
                self.done = True

    # For show and call on_mouse function
    def run(self,image):
        # Let's create our working window and set a mouse callback to handle events
        #image = cv2.resize(image, (224, 224)) 
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            cv2.imshow(self.window_name, image)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True
        cv2.destroyWindow(self.window_name)
        return image
