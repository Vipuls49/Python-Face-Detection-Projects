import cv2
import numpy as np

#define function to track object
def start_tracking():
    #initialize video capture object
    cap=cv2.VideoCapture(0)
    
    #define scaling factor
    scaling_factor=0.40
    
    #no of frames to track
    num_frames_to_track=5
    
    #skipping factor
    num_frames_jump=2
    
    #initialize variables
    tracking_paths=[]
    frame_index=0
    
    #define tracking parameters
    tracking_params=dict(winSize=(11,11),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
    
    #iterate until esc key
    while True:
        
        #capture current frame
        _,frame=cap.read()
        
        #resize frame
        frame=cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
        
        #convert to grayscale
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        #create copy of frame
        output_img=frame.copy()
        
        if len(tracking_paths)>0:
            #get images
            prev_img,current_img=prev_gray,frame_gray
            
            #organize feature points
            feature_points_0=np.float32([tp[-1] for tp in tracking_paths]).reshape(-1,1,2)
            
            #compute optical flow
            feature_points_1,_, _=cv2.calcOpticalFlowPyrLK(prev_img,current_img,feature_points_0,None,**tracking_params)
            
            #compute reverse optical flow
            feature_points_0_rev,_,_=cv2.calcOpticalFlowPyrLK(current_img,prev_img,feature_points_1,None,**tracking_params)
            
            #compute diff between forward and reverse optical flow
            diff_feature_points=abs(feature_points_0 - feature_points_0_rev).reshape(-1,2).max(-1)
            
            good_points=diff_feature_points <1
            
            #initialize variable
            new_tracking_paths=[]
            
            #iterate through good feature points
            for tp,(x,y), good_points_flag in zip(tracking_paths,feature_points_1.reshape(-1,2),good_points):
                #if flag is not true then continue
                if not good_points_flag:
                    continue
                
                #append x and y coordinates
                tp.append((x,y))
                if len(tp) > num_frames_to_track:
                    del tp[0]
                    
                new_tracking_paths.append(tp)
                
                #draw circle 
                cv2.circle(output_img,(x,y),3,(0,255,0),-1)
                
            #update tracking paths
            tracking_paths=new_tracking_paths
            
            #draw lines
            cv2.polylines(output_img,[np.int32(tp) for tp in tracking_paths],False,(0,150,0))
            
        #after skipping right no of frames
        if not frame_index % num_frames_jump:
            #create mask and draw circles
            mask=np.zeros_like(frame_gray)
            mask[:]=255
            for x,y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask,(x,y),6,0,-1)
                
            #compute good feature track
            feature_points=cv2.goodFeaturesToTrack(frame_gray,mask=mask,maxCorners=500,qualityLevel=0.3,minDistance=7,blockSize=7)
            
            #check if feature points exist
            if feature_points is not None:
                for x,y in np.float32(feature_points).reshape(-1,2):
                    tracking_paths.append([(x,y)])
                    
        #update variables
        frame_index +=1
        prev_gray=frame_gray
        
        cv2.imshow('optical flow',output_img)
        
        #esc key
        c=cv2.waitKey(1)
        if c==27:
            break
        
if __name__=='__main__':
     start_tracking()
     
     cv2.destroyAllWindows()
        
        
        


