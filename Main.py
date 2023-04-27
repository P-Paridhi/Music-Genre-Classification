import tkinter as tk
import tkinter.filedialog
import librosa
import numpy as np
from keras.models import load_model
from tkinter import *
from tkinter import *
from tkinter import ttk
import tkinter
from PIL import Image, ImageTk
import pygame

# Load the saved model
model = load_model('model.h5')

# input shape
n_mfcc = 20
input_shape = (n_mfcc, 1293)


# function to extract the MFCC features from a file
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] >= input_shape[1]:
        mfccs = mfccs[:, :input_shape[1]]
    else:
        pad_width = input_shape[1] - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    return mfccs.reshape(input_shape)


# function to predict the genre of a music file
def predict_genre(file_path):
    # Load the trained model
    model = load_model('model.h5')

    # Load and preprocess the audio data
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=20)
    mfccs = np.expand_dims(mfccs, axis=-1)

    # prediction with the model
    prediction = model.predict(np.array([mfccs]))

    # genre label
    predicted_label = np.argmax(prediction, axis=1)
    genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']
    predicted_genre = genre_labels[predicted_label[0]]

    return predicted_genre


# Select File button
def select_file():
    global file_path
    file_path = tkinter.filedialog.askopenfilename()
    genre_label.configure(text=predict_genre(file_path))


# Play button
def play_file():
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


# Tkinter GUI
root = tk.Tk()
root.title("Music Genre Prediction")
root.geometry("2000x2000")

# Load background image
bg_img = Image.open("BG_image.jpg").resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_img_tk = ImageTk.PhotoImage(bg_img)
bg_label = Label(root, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

root.title("Music Genre Prediction")
root.geometry("1000x500")

# label
label = tk.Label(
    root, height=2, width=20,
    text='Music Genre Classification',
    font=("Helvetica", 24), fg='white', bg='black')

label.pack(ipadx=10, ipady=10)



# Select File button
select_button = tk.Button(root, height=3, width=10, text="Upload File", fg='white', bg='black', command=select_file)
select_button.pack()

# Pause button
def pause_music():
    pygame.mixer.music.pause()

pause_button = tk.Button(root, height=3, width=10, text="Pause", fg='white', bg='black', command=pause_music)
pause_button.pack()

# Play button
play_button = tk.Button(root, height=3, width=10, text="Play", fg='white', bg='black', command=play_file)
play_button.pack()

# genre label
genre_label = tk.Label(root, text="",height=2,width=20,bg = 'black',fg='white')
genre_label.pack()

# Start the GUI
root.mainloop()
