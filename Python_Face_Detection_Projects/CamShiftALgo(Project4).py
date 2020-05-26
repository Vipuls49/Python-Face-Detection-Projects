import  cv2
import numpy as np

#define class to handle object
class ObjectTracker(object):
    def __init__(self,scaling_factor=0.5):
        #initialize video capture object
        self.cap=cv2.VideoCapture(0)
        
        #capture frame from webcam
        _,self.frame=self.cap.read()
        
        #scaling factor of frame
        self.scaling_factor=scaling_factor
        
        #resize frame
        self.frame=cv2.resize(self.frame,None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
        
        #create window to display frame
        cv2.namedWindow('Object Tracker')
        
        #set mouse callback funtion
        cv2.setMouseCallback('Object Tracker',self.mouse_event)
        
        #rectangular region selection
        self.selection=None
        
        #starting pos
        self.drag_start=None
        
        #state of tracking
        self.tracking_state=0
        
    #define method to mouse events
    def mouse_event(self, event, x, y, flags, param):
        #convert x and y to 16 bit numpy int
        x,y=np.int16([x,y])
        
        #check if mouse button down event
        if event==cv2.EVENT_LBUTTONDOWN:
            self.drag_start=(x,y)
            self.tracking_state=0
            
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                
            
                #extract dimension frame
                h,w=self.frame.shape[:2]
            
                #get initial position
                xi,yi=self.drag_start
            
                #get min and max values
                x0,y0=np.maximum(0,np.minimum([xi,yi],[x,y]))
                x1,y1=np.minimum([w,h],np.maximum([xi,yi],[x,y]))
            
                #reset selection variable
                self.selection=None
                
                #finalize rectangular selection
                if x1-x0 >0 and y1-y0 >0:
                    self.selection=(x0,y0,x1,y1)
                
            else:
                #if selection is done start track
                self.drag_start=None
                if self.selection is not None:
                    self.tracking_state=1
                
    #method to start tacking object
    def start_tracking(self):
        #iterate until esc key pressed
        while True:
            #capture from webcam
            _,self.frame=self.cap.read()
            
            #resize input frame
            self.frame=cv2.resize(self.frame,None,fx=self.scaling_factor,fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
            
            #copy of frame
            vis=self.frame.copy()
            
            #convert frame to HSV colorspace
            hsv=cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
            
            #create mask
            mask=cv2.inRange(hsv,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
            
            #check if user selected region
            if self.selection:
                #extract coordinates of selected rectangle
                x0,y0,x1,y1=self.selection
                
                #extract tracking window
                self.track_window=(x0,y0,x1-x0,y1-y0)
                
                #extract region of intrest
                hsv_roi=hsv[y0:y1,x0:x1]
                mask_roi=mask[y0:y1,x0:x1]
                
                #compute histogram region of intrest
                hist=cv2.calcHist([hsv_roi],[0],mask_roi,[16],[0,180])
                
                #normalize and reshpae histogram
                cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
                self.hist=hist.reshape(-1)
                
                #extract region of interest
                vis_roi=vis[y0:y1,x0:x1]
                
                #compute image -ve
                cv2.bitwise_not(vis_roi,vis_roi)
                vis[mask == 0] =0
                
            #check if in tracking mode
            if self.tracking_state == 1:
                #reset selection
                self.selection=None
                
                #compute histogram back projection
                hsv_backproj=cv2.calcBackProject([hsv],[0],self.hist,[0,180],1)
                
                #compute citwise and bteween histogram
                
                hsv_backproj &=mask
                
                #defint termination
                term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
                
                #apply camshift on hsv_back
                track_box,self.track_window =cv2.CamShift(hsv_backproj,self.track_window,term_crit)
                
                #draw ellipse
                cv2.ellipse(vis,track_box,(0,225,0) ,2)
                
            #show output live
            cv2.imshow('Object Tracker',vis)
            
            c=cv2.waitKey(5)
            if c==27:
                break
            
        cv2.destroyAllWindows()
    
if __name__=='__main__':
    ObjectTracker().start_tracking()
    

