import os
import tkinter as tk
from tkinter import HORIZONTAL, filedialog, Canvas
from PIL import Image, ImageTk
import cv2

loadvideo = None
video = None
framecurrente = None
framezero = None
tk_image = None

windowvideo = tk.Tk()
windowvideo.geometry('1000x1000')

def loadvideoF():
    global loadvideo, video
    loadvideo = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4 *.mov *.mkv')])
    video = cv2.VideoCapture(loadvideo)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    trackframes.config(to=total_frames)
    framezerof()

def framezerof():
    global video, tk_image
    ret, frame = video.read()
    if ret:
        display_frame(frame)

def display_frame(frame):
    global tk_image
    # Resize the frame
    xr, yr = frame.shape[1], frame.shape[0]
    ratio = min(930/xr, 930/yr)
    xr = int(xr * ratio)
    yr = int(yr * ratio)
    frame = cv2.resize(frame, (xr, yr))
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the Image object into a TkPhoto object
    tk_image = ImageTk.PhotoImage(image=Image.fromarray(rgb_image))
    canvasvideo.create_image((930-xr)//2, 0, image=tk_image, anchor=tk.NW)
    canvasvideo.update()

def on_scale(val):
    global video
    frame_number = int(val)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    if ret:
        display_frame(frame)
def saveframeF():
    global trackframes, video
    video.set(cv2.CAP_PROP_POS_FRAMES, int(trackframes.get()))
    ret, frame = video.read()
    if ret:
        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the NumPy array to a PIL Image object
        pil_image = Image.fromarray(rgb_image)
        # Save the image
        pil_image.save(f"frame{int(trackframes.get())}_{os.path.splitext(os.path.basename(loadvideo))[0]}.png")
        print(f"frame {int(trackframes.get())} salvato")
          
buttonload = tk.Button(windowvideo, text="load video", command=loadvideoF)
buttonload.grid(row=0,column=0)
canvasvideo = Canvas(windowvideo, width=930, height=930,bg="green")
canvasvideo.grid(row=1,column=2)
trackframes = tk.Scale(windowvideo, length=500, from_=0, to=1000, orient=tk.HORIZONTAL, command=on_scale)
trackframes.grid(row=0, column=1,padx=10)
saveframe= tk.Button(windowvideo,text="Salva fotogramma", command= saveframeF)
saveframe.grid(row=0,column=2,padx=10)

windowvideo.mainloop()
