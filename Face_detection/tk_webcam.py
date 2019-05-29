from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
import numpy as np
import torch

## new
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

classes = ['Anger', 'Disgust', 'Happiness', 'Neutral', 'Surprise', 'Sadness']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
######

##class definition
import torch.nn as nn
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Wraining on CPU ...')
else:
    print('CUDA is available!  Working on GPU ...')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 128x128x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        # convolutional layer (sees 64x64x4 tensor)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        # convolutional layer (sees 32x32x8 tensor)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        # max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(4,4)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(8 * 8 * 16, 500)
        # linear layer (500 -> 6)
        self.fc2 = nn.Linear(500, 6)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        # add sequence of convolutional and max pooling layers
        #self.conv1(x)
        x = self.pool(F.relu(self.conv1(x)))
        #x.shape
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with softmax activation function
        x = self.softmax(self.fc2(x))
        #x = self.fc2(x)
        return x
      
model = Net()
print (model)
if train_on_gpu:
    model.cuda()

model_path = "emotion_rec_V2_CPU.pt"
modelA = Net()
#modelA.load_state_dict(torch.load(model_path))
modelA.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: storage))
if train_on_gpu:
    modelA.cuda()
print(modelA)

class DataObj(object):
    
    def __init__(self):self.out = [[0,0,0,0,0,0]]
    # simulate values
    
    def getAnger(self): return self.out[0][0]
    def getDisgust(self): return self.out[0][1]
    def getHappiness(self): return self.out[0][2]
    def getNeutral(self): return self.out[0][3]
    def getSurpricse(self): return self.out[0][4]
    def getSadness(self): return self.out[0][5]

    def set(self,output):self.out = output

servo = DataObj()

class Example(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, width=500, height= 150, background="black")
        self.canvas1 = tk.Canvas(self, width=250, height= 150, background="black")
        self.canvas1.pack(side="left", fill="both", expand=True)
        self.canvas.pack(side="right", fill="both", expand=True)

        # create lines for Emotions
        self.Anger_line = self.canvas.create_line(0,25,0,25, fill="red")
        self.canvas1.create_text(50, 12, fill = 'white', text="Anger")
        self.Disgust_line = self.canvas.create_line(0,50,0,50, fill="blue")
        self.canvas1.create_text(50, 37, fill = 'white', text="Disgust")
        self.Happiness_line = self.canvas.create_line(0,75,0,75, fill="green")
        self.canvas1.create_text(50, 62, fill = 'white', text="Happiness")
        self.Neutral_line = self.canvas.create_line(0,100,0,100, fill="orange")
        self.canvas1.create_text(50, 87, fill = 'white', text="Neutral")
        self.Surprise_line = self.canvas.create_line(0,125,0,125, fill="white")
        self.canvas1.create_text(50, 112, fill = 'white', text="Surprise")
        self.Sadness_line = self.canvas.create_line(0,150,0,150, fill="purple")
        self.canvas1.create_text(50, 137, fill = 'white', text="Sadness")

        self.anger = self.canvas1.create_text(180, 12, fill = 'white', text = '0' )
        self.disgust = self.canvas1.create_text(180, 37, fill = 'white', text = '0' )
        self.happiness = self.canvas1.create_text(180, 62, fill = 'white', text = '0' )
        self.neutral = self.canvas1.create_text(180, 87, fill = 'white', text = '0' )
        self.surprice = self.canvas1.create_text(180, 112, fill = 'white', text = '0' )
        self.sadness = self.canvas1.create_text(180, 137, fill = 'white', text = '0' )

        # start the update process
        self.update_plot()

    def update_plot(self):
        an = servo.getAnger()
        di = servo.getDisgust()
        ha = servo.getHappiness()
        ne = servo.getNeutral()
        su = servo.getSurpricse()
        sa = servo.getSadness()

        self.add_point(self.Anger_line, 25 - (servo.getAnger() * 25.0))
        self.add_point(self.Disgust_line, 50 - (servo.getDisgust() * 25.0))
        self.add_point(self.Happiness_line, 75 - (servo.getHappiness() * 25.0))
        self.add_point(self.Neutral_line, 100 - (servo.getNeutral() * 25.0))
        self.add_point(self.Surprise_line, 125 - (servo.getSurpricse() * 25.0))
        self.add_point(self.Sadness_line, 150 - (servo.getSadness() * 25.0))

        self.update_textfiel(self.anger,round(an*100,3))
        self.update_textfiel(self.disgust,round(di*100,3))
        self.update_textfiel(self.happiness,round(ha*100,3))
        self.update_textfiel(self.neutral,round(ne*100,3))
        self.update_textfiel(self.surprice,round(su*100,3))
        self.update_textfiel(self.sadness,round(sa*100,3))

        self.canvas.xview_moveto(1.0)
        self.after(100, self.update_plot)

    def update_textfiel(self, item, new_text):
        self.canvas1.itemconfigure(item, text=new_text)

    def add_point(self, line, y):
        coords = self.canvas.coords(line)
        x = coords[-2] + 1
        coords.append(x)
        coords.append(y)
        coords = coords[-800:] # keep # of points to a manageable size
        self.canvas.coords(line, *coords)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


##

def quit_(root):
    root.destroy()

def update_image(image_label, cam):
    (readsuccessful, f) = cam.read()
    gray_im = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    ##new
    faces = face_cascade.detectMultiScale(
	    gray_im,
	    scaleFactor = 1.1,
	    minNeighbors = 5,
	    minSize=(35, 35)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(f, (x,y), (x+w, y+h), (255, 0, 0),1)
        roi_color = f[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(128,128))
        gray = [[gray]]
        gray = np.array(gray)
        gray = torch.from_numpy(gray)
        gray = gray.float()
        ##print (gray)
        out = modelA(gray)
        out = out.detach().numpy()

        servo.set(out)

        print(classes[0],out[0][0],classes[1],out[0][1],classes[2],out[0][2],classes[3],out[0][3],classes[4],out[0][4],classes[5],out[0][5])
    #####
    ####
    a = Image.fromarray(f) ##changed here
    b = ImageTk.PhotoImage(image=a)
    image_label.configure(image=b)
    image_label._image_cache = b  # avoid garbage collection
    root.update()


def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    try:
        fps = int(float(count_of_deltas) / sum_of_deltas)
    except ZeroDivisionError:
        fps = 0
    fps_label.configure(text='FPS: {}'.format(fps))


def update_all(root, image_label, cam, fps_label):
    update_image(image_label, cam)
    update_fps(fps_label)
    root.after(20, func=lambda: update_all(root, image_label, cam, fps_label))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Project") 
    image_label = tk.Label(master=root)# label for the video frame
    image_label.pack()
    cam = cv2.VideoCapture(0) 
    fps_label = tk.Label(master=root)# label for fps
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    ####
    Example(root).pack(side="top", fill="both", expand=True)
    # quit button
    quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root))
    quit_button.pack()
    # setup the update callback
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label))
    root.mainloop()
