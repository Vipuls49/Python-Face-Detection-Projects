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
    
    #define scaling factor
    scaling_factor=0.5
    
    #reading frames from webcam
    #until esc key pressed
    
    while True:
        #grab current frame
        frame=get_frame(cap, scaling_factor)
        
        #convert image to HSV colorspace
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #define range of skin color in HSV
        lower=np.array([0,70,60])
        upper=np.array([50,150,255])
        
        #threshold HSV image to get only skin color
        mask=cv2.inRange(hsv, lower, upper)
        
        #bitwise and between mask and original image
        img_bitwise_and=cv2.bitwise_and(frame, frame,mask=mask)
        
        #run median blurring
        img_median_blurred=cv2.medianBlur(img_bitwise_and, 5)
        
        #display input and output
        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)
        
        #check if user press esc key
        c=cv2.waitKey(5)
        if c==27:
            break
        
    #close windows
    cv2.destroyAllWindows()


