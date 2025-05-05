# Red Color Image Detection using OpenCV, Matplotlib, and NumPy

## AIM:
 Highlights largest red object in the image and highlights it with green ellipse. It's then saved with object 2.jpg
 
## LIBRARIES:
- Opencv: used for image processing, object detection and contour manipulation
- Numpy: generally used for handling array (in project it was used for creating mask manipulating pixel data)
- Matplotlib : generally used for viasulising images and data (plt.imshow used to display images)

## Functions & working:
- Show() : display processed image using matplotlib
- Overlay_mask () : converts binary mask to RGB and overlays it on original image 
- find_biggest_contour () : find all contours in image and largest one by area. Then draws largest contour on call blank mask
- circle_contour (): fits an ellipse around biggest contour 
- find_object () : converts input to RGB format and rescale for uniform distribution
- image_blur : apply Gaussian blur to reduce noise and smooth image & image_blur_hsv : converts blurred image to hsv 

 Why hsv? -> Color segmentation is more effective.
(Hue, Saturation, Value) :: hue (color), saturation (intensity), and value (brightness)

- min red max red defined : define color range
- mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) & mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel) : apply morphological operations to clean mask
- identify largest contour and overlay clean mask.
- circle largest object within ellipse & then processed image saved as (object name)2.jpg.

## CHALLENGES :
While the project efficiently detected red objects in most images - it faced challenges when multiple red objects were present. 
This was due to overlapping objects or varying shades of red that made filteration difficult. 
However, the detection was accurate where objects were distinct.

## EFFECTIVENESS :
95% of accuracy in a dataset of 100 images.
 
# CONCLUSION:
This project serves the purpose of introduction to image processing and efficiently filteration of largest among red objects.



