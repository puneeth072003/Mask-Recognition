# **Mask-recognition**  
### **Table of Contents**  
-   [About Project](#project-mask-recognition)
-   [Prequisites](#prequisites)
-   [Output Preview](#here-is-a-preview)
-   [How to Contribute to the Project](#how-to-contribute-to-the-project)
-   [Feedback](#feedback)
-   [Contributors](#contributors)

## **Project Mask-recognition**
 Its a simple project in which using **Haar-cascade frontal face detection** algorithm, the face is recoginzed and marked in the given video uploaded or captured via web cam and tells whether the person is wearing mask or not.
 
   
 ![Facemask-Detection-Blog](https://user-images.githubusercontent.com/119479391/211579312-520bf0bb-80a2-4b0b-bdc8-079149fd42a5.jpg)
 
 ## **Prequisites**

- Firstly download any **IDE** to run the code.
- Download [**Pre-trained model**](/Prerequisite/Model.h5) and [**haarcascade_frontalface.xml**](/Prerequisite/haarcascade_frontalface.xml) both files from our repository**(Prerequsite folder)** to your the local machine.

- Copy the absolute paths of the downloaded prerequsite files and paste it in `model=load_model('C:\\Users\\Puneeth\\Desktop\Mask-Recognition\Prerequisite\Model.h5')` and `face_cascade=cv2.CascadeClassifier('C:\\Users\\Puneeth\\Desktop\Mask-Recognition\Prerequisite\haarcascade_frontalface.xml')` sections.  
**Note:** Above commands are just added for the reference and the content inside it will not be same for every computer.

- Download all the modules required to run the code which are mentioned below.
    - **Tensorflow** *which can be downloded by running below command in the terminal.*
        ```python 
            pip install tensorflow
        ```
        
    - **Numpy** *which can be downloded by running below command in the terminal.*
        ```python 
            pip install numpy
        ```
    - **Cv2** *which can be downloded by running below command in the terminal.*
        ```python 
            pip install cv2
        ```
    - **Keras** *which can be downloded by running below command in the terminal.*
        ```python 
            pip install keras
        ```
- Copy the **Source code** from the above src folder and paste it in the IDE and this project can be run in **2 ways** which are mentioned below.  
    - **Method 1:** Acessing the video in your machine.    
       For this method you have to **remove** the below mentioned line of code from the source code copied and add **absolute path** of the video in this `cap=cv2.VideoCapture('C:\\Users\\Puneeth\\Desktop\sample.mp4')` section.  
       
       ```python 
           cap=cv2.VideoCapture(0)
       ```    
  - **Method 2:** Directly by capturing the video via ***Web-cam***       
        In order to do so you have to **remove** the below mentioned line of code from the source code copied.  
       ```python 
            cap=cv2.VideoCapture('C:\\Users\\Puneeth\\Desktop\sample.mp4')
       ```

- After making the above changes run the code as usual.
- There you have it a working project in your local desktop.
- Feel free to Experiment with the Code and find what works best for you.

## Here is a Preview  
 - **Method 1: Output preview:**  
   ![output](https://user-images.githubusercontent.com/119479391/216803808-f4669f71-38f0-4bec-9a2d-61fd5de3991a.png)  
   
- **Method 2: Output preview:**  
   ![final](https://user-images.githubusercontent.com/119479391/216803822-30964f99-5c4e-46a6-8e7b-1bf88ba693b1.png)

**NOTE:** A separate window will pop up showing the output video sometimes it will run in background of the IDE insted of popping up so please monitor the changes when you run the program.

## How to contribute to the project
- Feel free to fork the project and try to understand how it works.
- If you find any **queries** feel free to raise a new issue.

## Feedback
- You can encourage me by giving ⭐ **Stars** on the project.
- I would be of great help if you 🚀 **Sponsor** me.

## Contributors
- [Puneeth Y @puneeth072003](https://github.com/puneeth072003)
