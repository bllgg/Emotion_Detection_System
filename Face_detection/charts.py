import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

   
Data1 = {'Country': ['US','CA','GER','UK','FR'],
        'GDP_Per_Capita': [45000,42000,52000,49000,47000]
       }

df1 = DataFrame(Data1, columns= ['Country', 'GDP_Per_Capita'])
df1 = df1[['Country', 'GDP_Per_Capita']].groupby('Country').sum()

root= tk.Tk() 
  

figure1 = plt.Figure(figsize=(6,5), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, root)
bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
df1.plot(kind='bar', legend=True, ax=ax1)
#ax1.set_title('Country Vs. GDP Per Capita')

root.mainloop()
