from __future__ import division
import tkinter as tk
from tkinter import *
import shutil
import csv
import numpy as np


from tkinter import *

from PIL import Image, ImageTk
import os
import glob
import random
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import cv2
import sys
import pytesseract
from PIL import Image, ImageOps
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
from PIL import Image
import pytesseract
import glob

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
        # Include tesseract executable in your path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Vgg16 utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess 

# Resnet utils 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# utils to dump 
from pickle import dump
# import glob
import glob
import cv2
import sys
import pytesseract
from PIL import Image, ImageOps
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from enum import Enum
import os, io
from google.cloud import vision
import pandas as pd

import mahotas


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"My First Project-35b542e48ed9.json"

client = vision.ImageAnnotatorClient()
# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256


global fd
fd= pd.DataFrame()

 
Name=False
Age=False
Address=False
Phone_no=False
Lot=False
Section=False
Grave=False



LARGE_FONT= ("Verdana", 12)

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


class TextExtractor(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)
        container.configure(background='black')
    
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


        
       
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)

        tk.Frame.configure(self,background='black')  
        label = tk.Label(self, text="Document Classifier and Text Extractor", bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        label1 = tk.Label(self, text="Operations Available", bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 25, 'italic bold underline'))
        label1.pack(pady=30,padx=10)

        button = tk.Button(self, text="Document Classifier",fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '),
                            command=lambda: controller.show_frame(PageOne))
        button.place(x=700,y=300)

        button2 = tk.Button(self, text="Text Extractor",fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '),
                            command=lambda: controller.show_frame(PageTwo))
        button2.place(x=1000,y=300)

        button4 = tk.Button(self, text="EXIT",fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '), command= self.quit
                     )
        button4.place(x=1200,y=700)


        
class PageOne(tk.Frame):

   def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background='black')       
        label = tk.Label(self, text="Document Classifier",bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        message10 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message10.place(x=700, y=600)
        message11 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message11.place(x=700, y=800)
        message12 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=50  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message12.place(x=700, y=400)
        def model():
            data_dir = "C:/Users/debpr/Downloads/forms/"
            X = []
            Y = []
            for img_path in glob.glob(data_dir+"*/*.jpg"):
                img = img_path.split("\\")
                label = None
                if img[-2] == "Type-1 blocks":
                    label = 1
                elif img[-2] == "Type-2 Empty":
                    label = 2
                elif img[-2] == "Type-3 lines":
                    label = 3
                elif img[-2] == "HandWritten":
                    label = 0
                else:
                    print("Unknown label type", img[-2])
                Y.append(label)
                X.append(img_to_array(load_img(img_path,target_size=(224, 224))))

                # extract features from each photo in the directory
                #load the model, extracting features from each photo
                # prepare image for VGG model, get features, Storing Features

            def extract_features_vgg(X):
                in_layer = Input(shape=(224, 224, 3))
                model = VGG16(include_top=False, input_tensor=in_layer)
                features = []
                for image in X:
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    feature = model.predict(image, verbose=0)
                    features.append(feature)
                return features


            X_vgg_features = extract_features_vgg(X)

            y_cat = tf.keras.utils.to_categorical(Y)

            def build_model(hidden_size = 128):
                import tensorflow.keras as keras
                global fnn_model
                fnn_model = keras.Sequential()
                fnn_model.add(keras.layers.Flatten())
                fnn_model.add(keras.layers.Dense(hidden_size, activation='relu'))
                fnn_model.add(keras.layers.Dropout(0.2))
                fnn_model.add(keras.layers.Dense(4, activation='softmax'))
                fnn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
                return fnn_model

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_vgg_features, Y, test_size = 0.2,random_state = 1212)

            fnn_model = build_model(hidden_size=128)
            fnn_model.fit(np.array(X_vgg_features), y_cat , epochs=10, batch_size=128,verbose =0)
            y_pred = fnn_model.predict(np.array(X_test))


            y_pred = np.argmax(y_pred, axis=1)
            from sklearn.metrics import classification_report, confusion_matrix
            #print('Confusion Matrix')
            #print(confusion_matrix(y_test, y_pred))
            #print('Classification Report')
            target_names = ['HandWritten','Type 1 Block', 'Type 2 Empty', 'type 3 Lines']
            #print(classification_report(y_test, y_pred, target_names=target_names))

            kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=1234)
            cvscores_vgg = []
            for train, test in kfold.split(X, Y):
                fnn_model = build_model(hidden_size=128)
                fnn_model.fit(np.array(X_vgg_features)[train], y_cat[train] , epochs=10, batch_size=128,verbose =0)
                score = fnn_model.evaluate(np.array(X_vgg_features)[test],y_cat[test],verbose=0)
                cvscores_vgg.append(score[1])
            message12.configure(text='Model Trained')
                    
                    
            #for i,score in enumerate(cvscores_vgg):
                #print("The accuracy for {} is {:.2f}".format(i+1,score))
                    
            #print("The average accuracy on 10-fold cross validation is {:.2f} with standard deviation {:.2f}"\.format(np.mean(cvscores_vgg),np.std(cvscores_vgg)))

        def img():
            image=[]
            image=(img_to_array(load_img(c,target_size=(224, 224))))
            def extract_features_vgg(image):
                    in_layer = Input(shape=(224, 224, 3))
                    model = VGG16(include_top=False, input_tensor=in_layer)
                    features = []
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    feature = model.predict(image, verbose=0)
                    features.append(feature)
                    return features


            X_vgg_features = extract_features_vgg(image)
            target_names = ['Type 1 Block', 'Type 2 Empty', 'type 3 Lines','HandWritten']
            global fnn_model
            prediction=fnn_model.predict(X_vgg_features )
            message11.configure(text=target_names[int(prediction[0][0])])

        def file_sel():
            global c
            path= filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            c= path
            message10.configure(text=c)
    



        dire = tk.Button(self, text="Build Model", command= model  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        dire.place(x=500, y=300)
        fil = tk.Button(self, text="Classify Document", command= img ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        fil.place(x=1100, y=300)
        fil = tk.Button(self, text="Select Document", command= file_sel ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        fil.place(x=800, y=300)
        back = tk.Button(self, text="Back", command=lambda: controller.show_frame(StartPage)  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        back.place(x=1600, y=800)


class PageTwo(tk.Frame):

   def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.configure(self,background='black')       
        label = tk.Label(self, text="Text Extractor",bg="black"  ,fg="yellow"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline'))
        label.pack(pady=10,padx=10)
        label = tk.Label(self, text="Multiple Images",bg="black"  ,fg="yellow"  ,width=30  ,height=3,font=('times', 20, 'italic bold underline'))
        label.place(x=550,y=150)
        label = tk.Label(self, text="Single Images",bg="black"  ,fg="yellow"  ,width=30  ,height=3,font=('times', 20, 'italic bold underline'))
        label.place(x=950,y=150)
        message = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=35  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message.place(x=500, y=400)
        message1 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=30  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message1.place(x=700, y=600)
        message2 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=55  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message2.place(x=1000, y=400)
        message3 = tk.Label(self, text="" ,bg="black"  ,fg="yellow"  ,width=30  ,height=2, activebackground = "black" ,font=('times', 15, ' bold ')) 
        message3.place(x=1000, y=600)

        def directory():
            global a
            directory = filedialog.askdirectory()
            a= directory
            message.configure(text= a)
        
        def file_select():
            global b
            path= filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            b= path
            global p_img
            #p_img=process_image_for_ocr(b)
            #mahotas.imsave('C:\AAI\Project_NDA\copy-1.jpg', p_img)
            message2.configure(text=b)
    

        
        
        IMAGE_SIZE = 1800
        BINARY_THREHOLD = 180
        global size
        size= None



        def process_image_for_ocr(file_path):
            
            temp_filename = set_image_dpi(file_path)
            im_new=remove_noise_and_smooth(temp_filename)
            return (im_new)

        


        def get_size_of_scaled_image(im):
            global size
            if size is None:
                length_x, width_y = im.size
                factor = max(1, int(IMAGE_SIZE / length_x))
                size = factor * length_x, factor * width_y
            return size


        def set_image_dpi(file_path):
            im = Image.open(file_path)
            im = ImageOps.expand(im, border=15)
            size = get_size_of_scaled_image(im)

            im_resized = im.resize(size, Image.ANTIALIAS)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_filename = temp_file.name
            im_resized.save(temp_filename, dpi=(300, 300))
            return temp_filename

        def image_smoothening(img):

            ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
            ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blur = cv2.bilateralFilter(th2,1,75,75)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return th3

        def remove_noise_and_smooth(file_name):
            img = plt.imread(file_name)
            filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41,3)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            img = image_smoothening(img)
            or_image = cv2.bitwise_or(img, closing)




            return or_image

        def text_extract():
            f=open("C:\\AAI\\Project_NDA\\output-1.txt", "a+")
            f.truncate(0)
            f.close()
            directory=a
            count=1
            for file in os.listdir(directory):
                
                filename = os.fsdecode(file)
                if filename.endswith(".jpg"):
                    image = os.path.join(directory, filename)
                    config=()
                    text = pytesseract.image_to_string((image))
                    f=open("C:\\AAI\\Project_NDA\\output-1.txt", "a+")
                    f.write("\n\n\n Document %d \n\n" %count)
                    f.write(text)
                    count=count+1    
                else:
                    continue
            message1.configure(text="Extraction completed")

        def get_document_bounds(response, feature):
            bounds=[]
            for i,page in enumerate(document.pages):
                for block in page.blocks:
                    if feature==FeatureType.BLOCK:
                        bounds.append(block.bounding_box)
                    for paragraph in block.paragraphs:
                        if feature==FeatureType.PARA:
                            bounds.append(paragraph.bounding_box)
                        for word in paragraph.words:
                            for symbol in word.symbols:
                                if (feature == FeatureType.SYMBOL):
                                    bounds.append(symbol.bounding_box)
                            if (feature == FeatureType.WORD):
                                bounds.append(word.bounding_box)
            return bounds 

        def draw_boxes(image, bounds, color,width=5):
            draw = ImageDraw.Draw(image)
            for bound in bounds:
                draw.line([
                    bound.vertices[0].x, bound.vertices[0].y,
                    bound.vertices[1].x, bound.vertices[1].y,
                    bound.vertices[2].x, bound.vertices[2].y,
                    bound.vertices[3].x, bound.vertices[3].y,
                    bound.vertices[0].x, bound.vertices[0].y],fill=color, width=width)
            return image

        def assemble_word(word):
            assembled_word=""
            for symbol in word.symbols:
                assembled_word+=symbol.text
            return assembled_word

        def find_word_location(document,word_to_find):
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            assembled_word=assemble_word(word)
                            if(assembled_word==word_to_find):
                                print( word.bounding_box)
                    

        def text_within(document,x1,y1,x2,y2): 
            text=""
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            for symbol in word.symbols:
                                min_x=min(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                                max_x=max(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                                min_y=min(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                                max_y=max(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                                if(min_x >= x1 and max_x <= x2 and min_y >= y1 and max_y <= y2):
                                    text+=symbol.text
                                    if(symbol.property.detected_break.type==1 or 
                                    symbol.property.detected_break.type==3):
                                        text+=' '
                                    if(symbol.property.detected_break.type==2):
                                        text+='\t'
                                    if(symbol.property.detected_break.type==5):
                                        text+='\n'
            
            return(text)
           
        def text_file_extract():

                    print(b[0:37])
                    f=open("C:\\AAI\\Project_NDA\\output-5.txt", "a+")
                    f.truncate(0)
                    f.close()
                    
                    with io.open( b,'rb') as image_file:

                        content = image_file.read()

                    # annotate Image Response

                    # construct an iamge instance
                    content_image = vision.types.Image(content=content)


                    response = client.document_text_detection(image=content_image)  # returns TextAnnotation

                    global document
                    document = response.full_text_annotation

                    bounds = get_document_bounds(response, FeatureType.WORD)
                    im=draw_boxes(Image.open(b), bounds, 'yellow')

                    bounds = get_document_bounds(response, FeatureType.BLOCK)
                    im=draw_boxes(Image.open(b), bounds, 'red')
                    render = ImageTk.PhotoImage(im)
                    img = Label(image=render)
                    img.image = render
                    img.place(x=0, y=0)

                    
                    message3.configure(text="Extraction completed")



                    
        def open_output():
            class LabelTool():
                def __init__(self, master):
                    # set up the main frame
                    self.parent = master
                    self.parent.title("LabelTool")
                    self.frame = Frame(self.parent)
                    self.frame.pack(fill=BOTH, expand=1)
                    self.parent.resizable(width = FALSE, height = FALSE)

                    # initialize global state
                    self.imageDir = ''
                    self.imageList= []
                    self.egDir = ''
                    self.egList = []
                    self.outDir = ''
                    self.cur = 0
                    self.total = 0
                    self.category = 0
                    self.imagename = ''
                    self.labelfilename = ''
                    self.tkimg = None

                    # initialize mouse state
                    self.STATE = {}
                    self.STATE['click'] = 0
                    self.STATE['x'], self.STATE['y'] = 0, 0

                    # reference to bbox
                    self.bboxIdList = []
                    self.bboxId = None
                    self.bboxList = []
                    self.hl = None
                    self.vl = None

            
                    self.mainPanel = Canvas(self.frame, cursor='tcross')
                    self.mainPanel.bind("<Button-1>", self.mouseClick)
                    self.mainPanel.bind("<Motion>", self.mouseMove)
                    self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
                    self.parent.bind("s", self.cancelBBox)
                    self.parent.bind("a", self.prevImage) # press 'a' to go backforward
                    self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

                    # showing bbox info & delete bbox
                    self.lb1 = Label(self.frame, text = 'Bounding boxes:')
                    self.lb1.grid(row = 1, column = 2,  sticky = W+N)
                    self.listbox = Listbox(self.frame, width = 22, height = 12)
                    self.listbox.grid(row = 2, column = 2, sticky = N)
                    self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
                    self.btnDel.grid(row = 3, column = 2, sticky = W+E+N)
                    self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
                    self.btnClear.grid(row = 4, column = 2, sticky = W+E+N)
                    self.checkbutton=Checkbutton(self.frame, text="Full Name",command=self.name).place(x=1000,y=300)
                    self.checkbutton=Checkbutton(self.frame, text="Age",command=self.age).place(x=1000,y=330)
                    self.checkbutton=Checkbutton(self.frame, text="Address",command=self.address).place(x=1000,y=360)
                    self.checkbutton=Checkbutton(self.frame, text="Phone Number",command=self.phone).place(x=1000,y=390)
                    self.checkbutton=Checkbutton(self.frame, text="Lot",command=self.lot).place(x=1000,y=420)
                    self.checkbutton=Checkbutton(self.frame, text="Section",command=self.section).place(x=1000,y=450) 
                    self.checkbutton=Checkbutton(self.frame, text="Grave Number",command=self.grave).place(x=1000,y=480)      
                    # control panel for image navigation
                    self.ctrPanel = Frame(self.frame)
                    self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
                    self.prevBtn = Button(self.ctrPanel, text='Save', width = 10, command = self.prevImage)
                    self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)



                    # display mouse position
                    self.disp = Label(self.ctrPanel, text='')
                    self.disp.pack(side = RIGHT)

                    self.frame.columnconfigure(1, weight = 1)
                    self.frame.rowconfigure(4, weight = 1)


                    


                    # set up output dir
                    self.outDir = ('C:\\AAI\\Project_NDA')
                    self.loadImage()

                def loadImage(self):
                    # load image
                    imagepath = b
                    self.img = Image.open(imagepath)
                    self.tkimg = ImageTk.PhotoImage(self.img)
                    self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
                    self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

                    # load labels
                    self.clearBBox()
                    self.imagename = os.path.split(imagepath)[-1].split('.')[0]
                    labelname = self.imagename + '.txt'
                    self.labelfilename = os.path.join(self.outDir, labelname)
                    bbox_cnt = 0
                    if os.path.exists(self.labelfilename):
                        with open(self.labelfilename) as f:
                            for (i, line) in enumerate(f):
                                if i == 0:
                                    bbox_cnt = int(line.strip())
                                    continue
                                tmp = [int(t.strip()) for t in line.split()]
            ##                    print tmp
                                self.bboxList.append(tuple(tmp))
                                tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                                        tmp[2], tmp[3], \
                                                                        width = 2, \
                                                                        outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                                self.bboxIdList.append(tmpId)
                                self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(tmp[0], tmp[1], tmp[2], tmp[3]))
                                self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])


                def saveImage(self):
                    with open(self.labelfilename, 'w') as f:
                        f.write('%d\n' %len(self.bboxList))
                        for bbox in self.bboxList:
                            f.write(' '.join(map(str, bbox)) + '\n')
                    print ('Image No.  saved')


                def mouseClick(self, event):
                    if self.STATE['click'] == 0:
                        self.STATE['x'], self.STATE['y'] = event.x, event.y
                    else:
                        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
                        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
                        self.bboxList.append((x1, y1, x2, y2))
                        self.bboxIdList.append(self.bboxId)
                        self.bboxId = None
                        self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
                        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
                    self.STATE['click'] = 1 - self.STATE['click']

                def mouseMove(self, event):
                    self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
                    if self.tkimg:
                        if self.hl:
                            self.mainPanel.delete(self.hl)
                        self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
                        if self.vl:
                            self.mainPanel.delete(self.vl)
                        self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
                    if 1 == self.STATE['click']:
                        if self.bboxId:
                            self.mainPanel.delete(self.bboxId)
                        self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                                        event.x, event.y, \
                                                                        width = 2, \
                                                                        outline = COLORS[len(self.bboxList) % len(COLORS)])

                def cancelBBox(self, event):
                    if 1 == self.STATE['click']:
                        if self.bboxId:
                            self.mainPanel.delete(self.bboxId)
                            self.bboxId = None
                            self.STATE['click'] = 0

                def delBBox(self):
                    sel = self.listbox.curselection()
                    if len(sel) != 1 :
                        return
                    idx = int(sel[0])
                    self.mainPanel.delete(self.bboxIdList[idx])
                    self.bboxIdList.pop(idx)
                    self.bboxList.pop(idx)
                    self.listbox.delete(idx)

                def clearBBox(self):
                    for idx in range(len(self.bboxIdList)):
                        self.mainPanel.delete(self.bboxIdList[idx])
                    self.listbox.delete(0, len(self.bboxList))
                    self.bboxIdList = []
                    self.bboxList = []

                def prevImage(self, event = None):
                    self.saveImage()

                def name(self):
                    global Name
                    Name=not(Name)
                    print(Name)

                def age(self):
                    global Age
                    Age=not(Age)

                def address(self):
                    global Address
                    Address=not(Address)

                def phone(self):
                    global Phone_no
                    Phone_no=not(Phone_no)
                def lot(self):
                    global Lot
                    Lot=not(Lot)
                def section(self):
                    global Section
                    Section=not(Section)
                def grave(self):
                    global Grave
                    Grave=not(Grave)









            ##    def setImage(self, imagepath = r'test2.png'):
            ##        self.img = Image.open(imagepath)
            ##        self.tkimg = ImageTk.PhotoImage(self.img)
            ##        self.mainPanel.config(width = self.tkimg.width())
            ##        self.mainPanel.config(height = self.tkimg.height())
            ##        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

            if __name__ == '__main__':
                root = Toplevel()
                tool = LabelTool(root)
                root.resizable(width =  True, height = True)
                root.mainloop()







        def open_output1():

            os.startfile("C:\\AAI\\Project_NDA\\output-1.txt")
            message1.configure(text="")
        
        
            
       
       
        def database1():
            raw = []
            imagename = os.path.split(b)[-1].split('.')[0]
            labelname =  imagename + '.txt'
            with open('C:\AAI\Project_NDA\\'+str(labelname)) as f:
                for line in f:
                    raw.append(line.split())
            data = pd.DataFrame(raw)
            data=data.dropna()
            lis=[]
            for i in range(len(data)):
                rowData = data.iloc[i]
                x1=int(rowData.iloc[0])
                y1=int(rowData.iloc[1])
                x2=int(rowData.iloc[2])
                y2=int(rowData.iloc[3])
                text_within(document,x1,y1,x2,y2)
                lis.append(str(text_within(document,x1,y1,x2,y2)))
            
            lis_1=[]

            for i in range(len(lis)):
                if i==0 and Name==True:
                    lis_1.append(lis[0])
                elif i==0 and Name==False:
                    lis_1.append('Nan')
                if i==1 and Age==True:
                    lis_1.append(lis[1])

                elif i==1 and Age==False:
                    lis_1.append('Nan')
                if i==2 and Address==True:
                    lis_1.append(lis[2])
                elif i==2 and Address==False:
                    lis_1.append('Nan')

                if i==3 and Phone_no==True:
                    lis_1.append(lis[3])
                elif i==3 and Phone_no==False:
                    lis_1.append('Nan')
                if i==4 and  Lot==True:
                    lis_1.append(lis[4])

                elif i==4 and Lot==False:
                    lis_1.append('Nan')
            
                if i== 5 and Section==True:
                    lis_1.append(lis[5])
                    
                elif i==5 and Section==False:
                    lis_1.append('Nan')
                    
                if i==6 and Grave==True:
                    lis_1.append(lis[6])
            
                elif i==6 and Grave==False:
                    lis_1.append('Nan')

            print(lis_1)
            main_db=pd.Series(lis_1)
            main_db=main_db.transpose()
            global fd
            fd=fd.append(main_db,ignore_index=True)

            fd.to_excel('C:\\AAI\\Project_NDA\\output.xlsx')
            os.startfile("C:\\AAI\\Project_NDA\\output.xlsx")        








              


    
       
        text = tk.Button(self, text="Text Extractor", command= text_extract  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        text.place(x=700, y=500)

        back = tk.Button(self, text="Back", command=lambda: controller.show_frame(StartPage)  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        back.place(x=1600, y=800)

        dire = tk.Button(self, text="Select Directory", command= directory  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        dire.place(x=700, y=300)
        fil = tk.Button(self, text="Select File", command= file_select  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        fil.place(x=1100, y=300)
        ext = tk.Button(self, text="Text Extract ", command= text_file_extract  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        ext.place(x=1100, y=500)
        out = tk.Button(self, text="View Text", command= open_output  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        out.place(x=1100, y=700)
        out1 = tk.Button(self, text="View Text ", command= open_output1  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        out1.place(x=700, y=700)
        
        out2 = tk.Button(self, text="database ", command= database1  ,fg="yellow"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        out2.place(x=1100, y=900)
        
        

        





app = TextExtractor()
app.mainloop()