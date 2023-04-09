import tkinter as tk
import tkinter.ttk as ttk

import predict

from tkinter import filedialog, font
from ttkthemes import ThemedTk
from PIL import Image, ImageTk


class Styles(ttk.Style) :
    def __init__(self, **kwargs):
        super().__init__()
        
        # Label styles
        self.configure("text1.TLabel", background="black", foreground="white", font=("lucida",20))
        self.configure("text2.TLabel", background="black", foreground="white", font=("times",12))
        
        # Button styles
        self.configure("browse.TButton",font=("lucidabright",15),width=8,height=3,cursor="@hand2",anchor=tk.CENTER,background="dim gray",relief=tk.FLAT)
        self.map("browse.TButton",
            foreground=[('pressed', 'black'), ('active', 'white')],
            background=[('pressed', 'black'), ('active', 'gray')])


class Root(ThemedTk) :
    def __init__(self, **kwargs) :
        ThemedTk.__init__(self, theme="black", **kwargs)
        self.title("Brain Tumor Classifier")
        self.configure(bg = "black")
        self.geometry("650x610")
        self.resizable(0, 0)
        
        self.selected_file = ""
        
        self.setup_window()

    
    def display_img(self, image, img_label) :
        img = Image.open(image)
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)
        img_label.image = img
        
    
    
    def browse_files(self, selected, img_label, result_label) :
        filename = filedialog.askopenfilename(
            initialdir = "/",
            title = "Select a File",
            filetypes = (
                ("Jpeg Files", "*.jpg*"),
                ("PNG Files", "*.png*"),
                ("all Files", "*.*")
                )
            )
        
        self.selected_file = filename
        selected.configure(text=f"Opened: {filename}")
        
        # Display the image onto the screen
        self.display_img(filename, img_label)
        # Classify the image
        result_label.configure(text=f"Result: {predict.load_and_predict(filename)}")
    
    
    def setup_window(self) :
        label = ttk.Label(self, text="Browse system files:", style="text1.TLabel")
        selected = ttk.Label(self, text="", style="text2.TLabel")
        img_label = ttk.Label(self)
        result_label = ttk.Label(self, text=None, style="text1.TLabel")
        browse_button = ttk.Button(self, text="BROWSE", command=lambda: self.browse_files(selected, img_label, result_label), style="browse.TButton")
        
        # Placing the widgets onto the screen
        self.grid_columnconfigure(0, weight=1)
        label.grid(row=0, column=0, pady=30)
        browse_button.grid(row=1, column=0)
        selected.grid(row=2, column=0, pady=20)
        img_label.grid(row=3, column=0)
        result_label.grid(row=4, column=0, pady=30)



if __name__ == "__main__" :
    root = Root()
    style = Styles()
    root.mainloop()