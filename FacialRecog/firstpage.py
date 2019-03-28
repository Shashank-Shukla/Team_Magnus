try:
    from Tkinter import *
except ImportError:
    from tkinter import *
import os
import cv2
from datetime import datetime
import numpy as np
from PIL import Image
import time, sys, xlwrite
import firebase.firebase_ini as fire
root=Tk()
root.configure(background="white")

def createDB():
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    face_id=input('Enter your ID: ')
    vid_cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    assure_path_exists("dataset/")
    while(True):
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(200)
        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('frame', image_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif count>=50:
            print("Successfully Captured")

            break
    vid_cam.release()
    cv2.destroyAllWindows()


def trainDataSet():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    def getImagesAndLabels(path):
        #get the path of all the files in the folder
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        #create empth face list
        faceSamples=[]
        #create empty ID list
        Ids=[]
        #now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            #getting the Id from the image
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces=detector.detectMultiScale(imageNp)
            #If a face is there then append that in the list as well as Id of it
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
        return faceSamples,Ids
    faces,Ids = getImagesAndLabels('dataSet')
    s = recognizer.train(faces, np.array(Ids))
    print("Successfully trained!")
    recognizer.write('trainer/trainer.yml')

def function3():
    start=time.time()
    period=8
    face_cas = cv2.CascadeClassifier('haarcascade_profileface.xml')
    cap = cv2.VideoCapture(0);
    recognizer = cv2.face.LBPHFaceRecognizer_create();
    recognizer.read('trainer/trainer.yml');
    flag = 0;
    id=0;
    filename='filename';
    dict = {'item1': 1}
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cap.read();
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        faces = face_cas.detectMultiScale(gray);
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
            id,conf=recognizer.predict(roi_gray)
            if(conf < 200):
                 if(id==1):
                    id='Kunal'
                    if((str(id)) not in dict):
                        filename=xlwrite.output('attendance','class1',1,id,'P');
                        dict[str(id)]=str(id);

                 elif(id==2):
                    id = 'Anshu'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 2, id, 'P');
                        dict[str(id)] = str(id);

                 elif(id==3):
                    id = 'Divya'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 3, id, 'P');
                        dict[str(id)] = str(id);

                 elif(id==4):
                    id = 'AziroAzide'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 4, id, 'P');
                        dict[str(id)] = str(id);

                 else:
                     id = 'Unrecognized Person '+str(id)
                     filename=xlwrite.output('attendance','class1',5,id,'P');
                     flag+=1
                     break

            else:
                print("Failure in reading!!!!!")
            cv2.putText(img,str(id)+" "+str(conf),(x,y-10) ,font , 0.55, (120,255,120),1)
            #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
        cv2.imshow('frame',img);
        #cv2.imshow('gray',gray);
        if flag == 10:
            print("Transaction Blocked")
            break;
        if time.time()>start+period:
            break;
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break;

    cap.release();
    cv2.destroyAllWindows();


"""
def function5():
   os.startfile(os.getcwd()+"/developers/diet1frame1first.html");
"""

def function6():
    root.destroy()




def attend():
    os.startfile(os.getcwd()+"/firebase/attendance_files/attendance"+str(datetime.now().date())+'.xls')

root.title("SRM-IoT_HACK => TEAM MAGNUS")

Button(root,text="Create Dataset",font=("times new roman",20),bg="#000000",fg='white',command=createDB).grid(row=2,columnspan=2,sticky=E+N+W+S,padx=5,pady=5)

Button(root,text="Train Dataset",font=("times new roman",20),bg="#000000",fg='white',command=trainDataSet).grid(row=3,columnspan=2,sticky=N+E+S+W,padx=5,pady=5)

Button(root,text="Recog and background Attendance",font=('times new roman',20),bg="#000000",fg="white",command=function3).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

Button(root,text="Exit",font=('times new roman',20),bg="maroon",fg="white",command=function6).grid(row=9,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)


root.mainloop()
