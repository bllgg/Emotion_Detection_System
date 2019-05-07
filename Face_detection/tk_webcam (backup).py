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
    # quit button
    quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root))
    quit_button.pack()
    # setup the update callback
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label))
    root.mainloop()
