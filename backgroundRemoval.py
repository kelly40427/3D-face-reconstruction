import cv2
import numpy as np
import os
import glob

# Mouse event
def on_mouse(event,x,y,flag,param):        
    global rect
    global leftButtonDowm
    global leftButtonUp
    
    #left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False
        
    # mouse move
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and  not leftButtonUp:
            rect[2] = x
            rect[3] = y        
  
    # mouse up
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDowm and  not leftButtonUp:
            x_min = min(rect[0],rect[2])
            y_min = min(rect[1],rect[3])
            
            x_max = max(rect[0],rect[2])
            y_max = max(rect[1],rect[3])
            
            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDowm = False      
            leftButtonUp = True


#Get the folder
folder_name = 'subject1/subject1Left'
folder_path_input = os.path.join('input',folder_name)
file_list = os.listdir(folder_path_input)

# Read the first image
file_list = glob.glob(os.path.join(folder_path_input, '*.jpg'))
img = cv2.imread(file_list[0])

# Create an initial mask for the GrabCut algorithm
mask = np.zeros(img.shape[:2], np.uint8)

# Create models for the foreground and background
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = [0,0,0,0]

leftButtonDowm = False                    
leftButtonUp = True                       

cv2.namedWindow('img',cv2.WINDOW_NORMAL)                    
cv2.setMouseCallback('img',on_mouse)     
cv2.imshow('img',img)                    
rect2=[0,0,0,0]

while cv2.waitKey(2) == -1:
    if rect2!=[0,0,0,0]:
        break
    #left button down,draw rectangle
    if leftButtonDowm and not leftButtonUp:  
        img_copy = img.copy()
        cv2.rectangle(img_copy,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)  

        cv2.imshow('img',img_copy)
        
    #left button down,finish rectangle 
    elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        rect[2] = rect[2]-rect[0]
        rect[3] = rect[3]-rect[1]
        rect_copy = tuple(rect.copy())   
        print(rect_copy)
        rect2 = rect

        rect = [0,0,0,0]
        #segment the object
        cv2.grabCut(img,mask,rect_copy,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
            
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img_show = img*mask2[:,:,np.newaxis]
        #show the result
        cv2.namedWindow('grabcut',cv2.WINDOW_NORMAL) 
        cv2.imshow('grabcut',img_show)
        cv2.imshow('img',img)   

folder_path_output = os.path.join('output',folder_name)
for file_path_in in file_list:
    img_rb = cv2.imread(file_path_in)
    filename = os.path.basename(file_path_in)

    #segment the object
    cv2.grabCut(img_rb,mask,rect_copy,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_show = img_rb*mask2[:,:,np.newaxis]
    
    file_path_out = os.path.join(folder_path_output,filename)
    cv2.imwrite(file_path_out,img_show)
'''
'''
cv2.waitkey(2000)
cv2.destroyAllWindows()
