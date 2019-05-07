
import tkinter as tk
import random

class ServoDrive(object):
    def __init__(self,testMsg):print(testMsg)
    # simulate values
    def getAnger(self): return random.randint(0,25)
    def getDisgust(self): return random.randint(25,50)
    def getHappiness(self): return random.randint(50,75)
    def getNeutral(self): return random.randint(75,100)
    def getSurpricse(self): return random.randint(100,125)
    def getSadness(self): return random.randint(125,150)

servo = ServoDrive("TestMsg")

class Example(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        #self.servo = ServoDrive("TestMsg")
        self.canvas = tk.Canvas(self, width=500, height= 150, background="black")
        self.canvas1 = tk.Canvas(self, width=150, height= 150, background="black")
        self.canvas1.pack(side="left", fill="both", expand=True)
        self.canvas.pack(side="right", fill="both", expand=True)

        # create lines for velocity and torque
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

        self.anger = self.canvas1.create_text(120, 12, fill = 'white', text = '0' )
        self.disgust = self.canvas1.create_text(120, 37, fill = 'white', text = '0' )
        self.happiness = self.canvas1.create_text(120, 62, fill = 'white', text = '0' )
        self.neutral = self.canvas1.create_text(120, 87, fill = 'white', text = '0' )
        self.surprice = self.canvas1.create_text(120, 112, fill = 'white', text = '0' )
        self.sadness = self.canvas1.create_text(120, 137, fill = 'white', text = '0' )

        # start the update process
        self.update_plot()

    def update_plot(self):
        an = servo.getAnger()
        di = servo.getDisgust()
        ha = servo.getHappiness()
        ne = servo.getNeutral()
        su = servo.getSurpricse()
        sa = servo.getSadness()

        self.add_point(self.Anger_line, an)
        self.add_point(self.Disgust_line, di)
        self.add_point(self.Happiness_line, ha)
        self.add_point(self.Neutral_line, ne)
        self.add_point(self.Surprise_line, su)
        self.add_point(self.Sadness_line, sa)


        self.update_textfiel(self.anger,an)
        self.update_textfiel(self.disgust,di)
        self.update_textfiel(self.happiness,ha)
        self.update_textfiel(self.neutral,ne)
        self.update_textfiel(self.surprice,su)
        self.update_textfiel(self.sadness,sa)

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

if __name__ == "__main__":
    root = tk.Tk()
    Example(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
