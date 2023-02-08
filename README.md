# **Mask-recognition**  
### **Table of Contents**  
-   [About Project](#project-mask-recognition)
-   [Prequisites](#prequisites)
-   [How to train your own model](#how-to-train-your-own-model)
-   [Output Preview](#here-is-a-preview)
-   [How to Contribute to the Project](#how-to-contribute-to-the-project)
-   [Feedback](#feedback)
-   [Contributors](#contributors)

## **Project Mask-recognition**
 Its a simple project in which using **Haar-cascade frontal face detection** algorithm, the face is recoginzed and marked in the given video uploaded or captured via web cam and tells whether the person is wearing mask or not.
 
   
 ![Facemask-Detection-Blog](https://user-images.githubusercontent.com/119479391/211579312-520bf0bb-80a2-4b0b-bdc8-079149fd42a5.jpg)
 
 ## **Prequisites**

- Firstly download any **IDE** to run the code.
- Download [**Pre-trained model**](/Prerequisite/Model.h5) and [**haarcascade_frontalface.xml**](/Prerequisite/haarcascade_frontalface.xml) both files from our repository to your the local machine.  
**Note:** [Click here](https://github.com/puneeth072003/Mask-Recognition/tree/model#how-to-train-your-own-model) if you want to try build your own model insted of using a pre-trained one.

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
- Copy the [***Source code***](/src.py) from the above src folder and paste it in the IDE and this project can be run in **2 ways** which are mentioned below.  
    - **Method 1:** Acessing the video in your machine.    
       For this method you have to **remove** the below mentioned line of code from the source code copied and add **absolute path** of the video to be processed in this `cap=cv2.VideoCapture('C:\\Users\\Puneeth\\Desktop\sample.mp4')` section.  
       
       ```python 
           cap=cv2.VideoCapture(0)
       ```    
  - **Method 2:** Directly by capturing the video via ***Web-cam***       
        In order to do so you have to **remove** the below mentioned line of code from the source code copied.  
       ```python 
            cap=cv2.VideoCapture('C:\\Users\\Puneeth\\Desktop\sample.mp4')          # Alterations to be made here
       ```

- After making the above changes run the code as usual.
- There you have it a working project in your local desktop.
- Feel free to Experiment with the Code and find what works best for you.

## How to train your own model
- **Step 1:** Please **download** and upload the [***DataSets.zip***](/train_model/DataSets.zip) required to train your model from our repository to your **Google Drive**.
[*Click here to get a direct G-Drive link*](https://drive.google.com/file/d/16PKeI2RIz_r-JTqbDm6hfWQ-he7Kk8rV/view?usp=sharing)    
**Note:** You can use other data sets insted of this one but please note that the more you train your model more accurate will be its predictions.

- **Step 2:** Further go ahead and **open** your [***Google Colaboratory***](https://colab.research.google.com/) and create a new note book.
  
- **Step 3:** **Import** these modules given below.
    ```python 
       from tensorflow.keras.models import Sequential
       from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
       from tensorflow.keras.optimizers import Adam
       from tensorflow.keras.preprocessing.image import ImageDataGenerator
       import numpy as np
       import matplotlib.pyplot as plt     
    ```  
- **Step 4:** **Import** google drive from google.colab and mount it by using the following command.  
   ```python 
      from google.colab import drive          
      drive.mount('/content/drive/')   # Alteration to be made here
   ```
- **Step 5:** **Unzip** the downloaded zip file. In order to achive that command `!unzip` is used.  
  **Note:** Please to paste paths where you have uploaded the zip file in your drive have to be pasted below parenthesis.
  ```python
     !unzip -uq "/content/drive/MyDrive/project/face mask/archive" -d "/content/drive/MyDrive/project/face mask/archive"   # Alteration to be made here
  ```
- **Step 6:** In this step we **assign path names** to their respective complete paths making code more versatile and understandable instead of hard-coding every path name manually. This is done using the function `os.path.join()` as shown below.
  ```python
     import os 
     main_dir="/content/drive/MyDrive/project/face mask/archive/New Masks Dataset"   # Alteration to be made here
     train_dir=os.path.join(main_dir,'Train')    
     test_dir=os.path.join(main_dir,'Test')
     valid_dir=os.path.join(main_dir,'Validation')
     train_mask_dir=os.path.join(train_dir, 'Mask')       
     train_nomask_dir=os.path.join(train_dir, 'Non Mask')
  ```
- **Step 7:** Here, we take inputs of the original data and then **transform** it based on set of rules, returning the output resultant containing solely the newly changed data. This is done with the help of function `ImageDataGenerator()` as shown in the commands below and the function `.flow_from_directory()` is used just read out  batch_size, number of images and their associated labels.
  ```python
     train_datagen=ImageDataGenerator(rescale=1./255,
                                 zoom_range = 0.2,
                                 rotation_range = 40,
                                 horizontal_flip = True)             
     test_datagen=ImageDataGenerator(rescale=1./255)
     valid_datagen=ImageDataGenerator(rescale=1./255) 

     train_generator = train_datagen.flow_from_directory(train_dir,
                                                         target_size=(150,150),
                                                         batch_size=32,
                                                         class_mode='binary') 
     test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=(150,150),
                                                   batch_size=32,       
                                                   class_mode='binary')
     valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')
  ```
 - **Step 8:** In this step we basically **add** different layers to the model which is accomplished by the function `model.add()` and function `model.summary()` gives out the summary of the model.
 [*Click here to know more about the sequential model*](https://keras.io/guides/sequential_model/#:~:text=A%20Sequential%20model%20is%20appropriate,tensor%20and%20one%20output%20tensor.&text=A%20Sequential%20model%20is%20not,multiple%20inputs%20or%20multiple%20outputs)
    ```python
    model = Sequential() # we need this model to image conversions

    model.add(Conv2D(32,(3,3),padding='SAME',activation='relu', input_shape=(150,150,3)))  
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout (0.5))

    model.add(Conv2D(64, (3,3),padding='SAME', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Dropout (0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu')) 
    model.add(Dropout (0.5)) 
    model.add(Dense(1,activation='sigmoid')) 
    model.summary()
    ```
  
 - **Step 9:** Here we compile the model basically **finalise** the model and make it completely ready to further training purpose which is done by the function `model.compile()` as shown below.
    ```python
    model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy']) 
    ```
 
 - **Step 10:** Its is the most important and the most **time consuming part** of the entire process. Here with the help of function `model.fit()` we **train** our model to improve its accuracy in predictions using the Data-sets downloaded.
    ```python
    history = model.fit(train_generator,
    epochs = 30, # epoches basically mean periods or sessions
    validation_data= valid_generator)
    ```
 
 - **Step 11:** In this step we **caluclate** and print out the total accuracy and loss of the model using the function `model.evaluate()`.
    ```python
    test_loss, test_acc = model.evaluate(test_generator)
    print('test aac :{} test loss:{}'.format(test_acc,test_loss))
    ```
 
 - **Step 12:** In this step, we basically **test** out our model's prediction using series of functions and print out wether the mask is present or absent. This can be done using the commands listed below.
    ```python
    import numpy as np 
    from google.colab import files 
    import keras.utils as image

    uploaded=files.upload()    
    #print (uploaded)
    for f in uploaded.keys():
        img_path= '/content/'+f 
        img = image.load_img(img_path, target_size=(150,150)) 
        images=image.img_to_array(img) # this is used to convert into a numpy array
        images=np.expand_dims(images,axis=0) # this is used to expand the dimensions of the array 
        prediction=model.predict(images)
        if prediction==0:
            print(f, 'mask is present')
        else:
            print(f,' no mask is present')
    ```
 - **Step 13:** It is the last and final step in this process here you basically **save** the model in your drive using the function `model.save()`.
    ```python
    model.save('Saved_model.h5')
    ```
 - **Step 14:** **Download** the Saved_model from your GoogleDrive to your local machine. So, that we can further use the model in the Project.
**NOTE:** You can get the refrence copy of all the outputs and the code at [***model_training.ipynb***](/train_model/model_training.ipynb)

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
