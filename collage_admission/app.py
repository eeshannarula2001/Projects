
import numpy
from tkinter import *

labels = []
entries = []

import consts

root = Tk()

root.title("Check your chance")
root.geometry('400x400')

root.configure(background = "white");

for i in consts.features:
    a = Label(root ,text = i)
    a.grid(row = consts.features.index(i),column = 0)
    labels.append(a)

    b = Entry(root)
    b.grid(row = consts.features.index(i),column = 1)
    entries.append(b)

T = Text(root, height=2, width=30)
T.grid(row=12,column=1)

def clicked():

    input = [1]
    for i in entries:
        input.append(float(i.get()))
    input = numpy.array(input)
    theta = numpy.array(consts.theta)
    prediction = numpy.matmul(input,theta)
    T.insert(END, 'your chances are :' + str(prediction) + '%')

btn = Button(root ,text="Submit",command = clicked).grid(row=12,column=0)

root.mainloop()
