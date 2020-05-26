import cv2
import numpy as np

#define function to get frame from webcam
def get_frame(cap,scaling_factor):
    #read current frame from video capture object
    _, frame=cap.read()
    
    #resize image
    frame=cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
    
    return frame

if __name__=='__main__':
    #define video capture object
    cap=cv2.VideoCapture(0)
    
    #define background subtractor
    bg_subtractor=cv2.createBackgroundSubtractorMOG2()
    
    #define no of previous frames it controls learning rate of algo
    #higher value of history indicates slow learning rate
    
    history=100
    
    #define learning rate
    learning_rate=1.0/history
    
    #reading frame from webcam  until esc key pressed
    while True:
        #grab current frame
        frame=get_frame(cap, 0.5)
        
        #compute mask
        mask = bg_subtractor.apply(frame, learningRate =learning_rate)
        
        #convert greyscale to rgb color
        mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        #display images
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask&frame)
        
        #check if esc key pressed
        c=cv2.waitKey(10)
        if c==27:
            break
    #release video capture
    cap.release()
    
    #close windows
    cv2.destroyAllWindows()
    
    
    

