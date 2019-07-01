import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import pickle

root = Tk()
root.geometry('1200x850')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('YÜZ TANIMA')
frame.config(background='white')
##label = Label(frame, text="YÜZ TANIMA ", bg='white', font=('Times 35 bold'))
##label.pack(side=TOP)
filename = PhotoImage(file="C:/Users/YEK/PycharmProjects/YuzveNesneTanima/Face/face.png")
background_label = Label(frame, image=filename)
background_label.pack(side=TOP)


def hel():
    help(cv2)


def exitt():
    exit()


def web():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Kamera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def webrec():
    capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('Sample1.avi', fourcc, 11.0, (640, 480))
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Kaydet', frame)
        op.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    op.release()
    capture.release()
    cv2.destroyAllWindows()


def webdet():

  face_cascade=cv2.CascadeClassifier('C:/Users/YEK/PycharmProjects/YuzveNesneTanima/Face/cascade/haarcascade_frontalface_alt2.xml')
  recognizer = cv2.face.LBPHFaceRecognizer_create()

  recognizer.read("trainner.yml")
  labels={"person_name":1}
  with open("labels.pickle", 'rb') as f:
      og_labels = pickle.load(f)

      labels = {v:k for k,v in og_labels.items()}
  capture = cv2.VideoCapture(0)
  while True:
      ret, frame = capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
      for (x, y, w, h) in faces:
          #print(x, y, w, h)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = frame[y:y + h, x:x + w]

          id_, conf = recognizer.predict(roi_gray)

          if  conf >= 45:# and conf <=85:

              print(id_)
              print(labels[id_])
              font=cv2.FONT_HERSHEY_SIMPLEX
              name=labels[id_]
              color=(255,255,255)
              stroke=2
              cv2.putText(frame, name,(x + w, y + h), font, 1, color, stroke, cv2.LINE_AA)



          color = (255, 0, 0)
          stroke = 2



          cv2.rectangle(frame,(x,y),(x + w, y + h), color, stroke)


      cv2.imshow('Yuz Tanima', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  capture.release()
  cv2.destroyAllWindows()


def webdetRec():


   face_cascade = cv2.CascadeClassifier('C:/Users/YEK/PycharmProjects/YuzveNesneTanima/Face/cascade/haarcascade_frontalface_alt2.xml')
   recognizer = cv2.face.LBPHFaceRecognizer_create()

   recognizer.read("trainner.yml")
   labels = {"person_name": 1}

   with open("labels.pickle", 'rb') as f:

       og_labels = pickle.load(f)
       labels = {v: k for k, v in og_labels .items()}

   capture = cv2.VideoCapture(0)

   while True:
       ret, frame = capture.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
       for (x, y, w, h) in faces:
           # print(x, y, w, h)
           roi_gray = gray[y:y + h, x:x + w]
           roi_color = frame[y:y + h, x:x + w]

           id_, conf = recognizer.predict(roi_gray)

           if conf >= 45:  # and conf <=85:

               print(id_)
               print(labels[id_])
               font = cv2.FONT_HERSHEY_SIMPLEX
               name = labels[id_]
               color = (255, 255, 255)
               stroke = 2
               cv2.putText(frame, name, (x + w, y + h), font, 1, color, stroke, cv2.LINE_AA)

           img_item = "image.png"
           cv2.imwrite(img_item, roi_color )

           color = (255, 0, 0)
           stroke = 2

           cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

       cv2.imshow('Yuz Tanima & Kaydetme', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   capture.release()
   cv2.destroyAllWindows()






bttn1 = Button(frame, padx=5, pady=5,width=39 ,bg='white', fg='black',relief=FLAT,command=web,text='Kamerayı Aç',font=('Times 15 bold'))
bttn1.place(x=5, y=176)

bttn2=Button(frame,padx=5,pady=5,width=39, bg='white',fg='black',relief=FLAT,command=webrec,text='Kamerayı Aç ve Kaydet',font=('Times 15 bold'))
bttn2.place(x=5, y=246)

bttn3 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=FLAT, command=webdet,text='Yüz Tanıma', font=('Times 15 bold'))
bttn3.place(x=5, y=316)

bttn4 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=FLAT, command=webdetRec,text='Yüz Tanıma & Kaydet', font=('Times 15 bold'))
bttn4.place(x=5, y=386)


bttn6 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=FLAT, text='Çıkış', command=exitt,font=('Times 15 bold'))
bttn6.place(x=210, y=478)

root.mainloop()