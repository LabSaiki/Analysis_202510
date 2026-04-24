import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class FolderName:
    def __init__(self):
        pass

    def select_analysis_folders(self):
        root = tk.Tk()
        root.geometry("400x400")
        root.title("Select Analysis Folders")

        button_frame = ttk.Frame(root,width=400,height=50)
        button_frame.pack()

        text_frame = ttk.Frame(root,width=400,height=350)
        text_frame.pack()

        selected_dirs = []

        def add_directory():
            dir_path = filedialog.askdirectory(initialdir="C:/Users/tsaik/PythonCode")
            if dir_path not in selected_dirs:
                selected_dirs.append(dir_path)
                dir_label = ttk.Label(text_frame, text=dir_path, anchor="w")
                dir_label.pack(fill='x')

        def back_selection():
            selected_dirs.pop()
            text_frame.winfo_children()[-1].destroy()


        def reset_selection():
            selected_dirs.clear()
            for widget in text_frame.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.destroy()


        add_button = ttk.Button(button_frame, text="Select Folder", command=add_directory)
        add_button.pack(side="left", padx=10)
        back_button = ttk.Button(button_frame, text="Back", command=back_selection)
        back_button.pack(side="left", padx=10)
        reset_button = ttk.Button(button_frame, text="Reset", command=reset_selection)
        reset_button.pack(side="right", padx=10)

        root.mainloop()

        return selected_dirs
    
def test():
    FolderName().select_analysis_folders()

    
if __name__ == "__main__":
    test()