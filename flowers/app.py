import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename

import cv2
import consts
from keras.models import load_model

import DataHelper as dh
from PIL import Image, ImageTk


model = load_model('flowers66.h5')


root = Tk()

PATH = StringVar()
CATIGORY = StringVar()

def browsefunc():
    PATH.set(askopenfilename())
    img = ImageTk.PhotoImage(Image.open(PATH.get()))
    panel.configure(image=img)
    panel.image = img

def predict():
    if PATH.get() is not None:
       input = dh.make_read_for_input(PATH.get())
       prediction = dh.FromOneHot(model.predict(input))[0]
       CATIGORY.set('prediction is : ' + consts.CAT[prediction])


panel = Label(root)
panel.grid(row = 0,column = 0)

browsebutton = Button(root, text="choose a image", command=browsefunc)
browsebutton.grid(row = 1,column=0)

predictbutton = Button(root, text="predict the image", command=predict)
predictbutton.grid(row = 2,column=0)

prediction = Label(root, textvariable=CATIGORY)
prediction.grid(row = 3,column=0)

mainloop()
