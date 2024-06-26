import cv2
from matplotlib import pyplot as plt
import numpy as np


green=(0,255,0)                          #For the green circle

def show(image):                         #Printing the image

    plt.figure(figsize=(10,10))
    plt.imshow(image,interpolation='nearest')

def overlay_mask(mask,image):            #For adding the overlay to the image
    
    rgb_mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    img=cv2.addWeighted(rgb_mask,0.5,image,0.5,0)
    return img

def find_biggest_contour(image):         #To find the biggest contour in the image

    image=image.copy()
    contours,hierarchy=cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 

    contour_sizes=[(cv2.contourArea(contour),contour)for contour in contours]
    biggest_contour=max(contour_sizes,key=lambda x: x[0])[1]

    mask=np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask,[biggest_contour],-1,255,-1)
    return biggest_contour,mask

def circle_contour(image,contour):      #To circle the final contour with a green ellipse
    image_with_ellipse=image.copy()
    ellipse=cv2.fitEllipse(contour)

    cv2.ellipse(image_with_ellipse,ellipse,green,2,cv2.LINE_AA)
    return image_with_ellipse

    
def find_object(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)      #Converting image from BGR to RGB format
    
    max_dimension=max(image.shape)                   #Scaling the image into a proper size
    scale=700/max_dimension
    image=cv2.resize(image,None,fx=scale,fy=scale)

    image_blur=cv2.GaussianBlur(image,(7,7),0)       #Blurring the image 
    image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)
    
    min_red=np.array([0,100,80])                     #Filter by color 
    max_red=np.array([10,256,256])


    mask1=cv2.inRange(image_blur_hsv,min_red,max_red)#Creating a mask for intensity of colour

   
    min_red1=np.array([170,100,80])                  #Filter by brightness
    max_red1=np.array([180,256,256])

    mask2=cv2.inRange(image_blur_hsv,min_red1,max_red1)#Creating a mask for intensity of brightness

    mask=mask1+mask2

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)) #Structing the image using an ellipse
    mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask_clean=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

    big_object_contour,mask_object=find_biggest_contour(mask_clean) #Selecting the biggest object from the image

    overlay=overlay_mask(mask_clean,image)              #Overlaying the mask with an outline

    circled=circle_contour(overlay,big_object_contour)  #Circling the image 

    show(circled)

    bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)         

    return bgr

image=cv2.imread('rose.jpg')             #Inputing the image from the user               
result=find_object(image)                

cv2.imwrite('rose2.jpg',result)          #Output of the image
