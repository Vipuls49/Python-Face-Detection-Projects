import cv2

#compute the frame difference

def frame_diff(prev_frame,cur_frame,next_frame):
    #difference between current and next frame
    diff_frame1=cv2.absdiff(next_frame, cur_frame)
    
    #differnce between current and previous frame
    diff_frame2=cv2.absdiff(cur_frame, prev_frame)
    
    return cv2.bitwise_and(diff_frame1, diff_frame2)

#define a function to get the current frame from webcam

def get_frame(cap,scaling_factor):
    #read current frame from video capture object
    _, frame=cap.read()
    
    #resize the image
    
    frame=cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)

    grey=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    return grey

if __name__=='__main__':
    
    #define video capture object
    cap=cv2.VideoCapture(0)
    
    #define scaling factor of image
    scaling_factor=0.5
    
    #grab previous frame
    prev_frame=get_frame(cap, scaling_factor)
    
    #grab current frame
    cur_frame=get_frame(cap, scaling_factor)
    
    #grab next frame
    next_frame=get_frame(cap, scaling_factor)
    
    #keep reading till esc key is pressed
    while True:
        #display frame diff
        
        cv2.imshow('Object Movement', frame_diff(prev_frame, cur_frame, next_frame))
        
        #update variables
        prev_frame=cur_frame
        cur_frame=next_frame
        
        #grab next frame
        next_frame=get_frame(cap, scaling_factor)
        
        #check for esc key
        key=cv2.waitKey(10)
        if key==27:
            break
    cv2.destroyAllWindows()
    
    