import cv2   # importing CV2 module         
from tensorflow.keras.models import load_model   # This module facilitates us to load our pre-trained model
from tensorflow.keras.preprocessing.image import load_img , img_to_array   # this module helps in convertion of image to array
import numpy as np   # this module helps with expand the array

model=load_model('C:\\Users\\Puneeth\\Desktop\Mask-Recognition\Prerequisite\Model.h5')    # this allows us to load the pretrained model

img_width, img_height= 150,150  # these are the default values to be set

face_cascade=cv2.CascadeClassifier('C:\\Users\\Puneeth\\Desktop\Mask-Recognition\Prerequisite\haarcascade_frontalface.xml')  # this allows us to use the downloded algorithm

cap=cv2.VideoCapture('C:\\Users\\Puneeth\\Desktop\sample.mp4')   # Please paste the absolute path of your video inside parenthesis
#cap=cv2.VideoCapture(0)           # Uncomment this if you have to use your web-cam to capture video
img_count_full=0    # declaring a counting variable
 
font=cv2.FONT_HERSHEY_SIMPLEX    # setting a font style
org=(1,1)        # setting origin of the Text
class_label=''   # setting the class label
fontScale=1      # setting font scale
color=(255,0,0)  # setting the font colour
thickness=2      # setting the thickness of the font

while True:    # Continously checks whether the face is captured or not
    img_count_full +=1
    response , color_img = cap.read()             
    if response==False:   # Breaks the loop whenever a face is captured to continue with the furthur process
        break

    scale=50     # setting scale of the image captured                           
    width=int(color_img.shape[1]*scale/100)   # setting up image dimensions
    height=int(color_img. shape[0]*scale/100)   # setting up image dimensions
    dim=(width, height)
     
    color_img = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)   # Setting up the video preview after caluclations
    
    gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)   # creating the gray scale image
    
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)    # dectecting multi scale (if multiple are present)
    
    img_count=0     # declaring a another counting variable
    for(x,y,w,h) in faces:     # This loop is to put a outline around the face that is detected and is to be processed
            org=(x-10,y-10)    # Setting origin for the outline box
            img_count+=1       # incrementation of image count
            color_face=color_img[y:y+h, x:x+w]   # setting dimensions for the outline box           
            cv2.imwrite('input/%d%dface.jpg'%(img_count_full,img_count),color_face)    # Writing on the image captured
            img=load_img('input/%d%dface.jpg'%(img_count_full,img_count),target_size=(img_width,img_height))  # loading the image captured
            img=img_to_array(img)    # converting image to array
            img=np.expand_dims(img,axis=0)     # By using the numpy we expand the shape of an array.
            
            prediction=model.predict(img)      # Using the model to predict wether the person is wearing mask or not
            if prediction==0:      # depending on the prediction this condition names the outline as Mask          
                class_label="Mask"
                color=(0,0,255)
            else:                  # depending on the prediction this condition names the outline as No Mask
                class_label="No Mask"
                color=(0,0,255)


            cv2.rectangle(color_img,(x,y),(x+w,y+h),(255,0,0),3)      # this statment puts a outline around the face of the person 
            cv2.putText(color_img,class_label,org,font,fontScale, color, thickness,cv2.LINE_AA) # this statment puts the text on the image 
    
    cv2.imshow("Face Mask Detection",color_img)   # this statment is to show the image modifications
    if cv2.waitKey(1) & 0xFF==ord('q'):  # this statment allows us to wait till next image is ready to display
            break
            
cap.release()   # through this statment once our work is done, we release the resources
cv2.destroyAllWindows()    # to destroy all the windows once the process is complete
