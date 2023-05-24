import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch.nn as nn
from CNN import CNN
from krzysiek.FCNN.NeuralNet import NeuralNet
import torch
from torchvision import models

selected_image = None
original_image = None
found_faces = []
coords = []


def cnn_button_click():
    if found_faces is None:
        print("No image selected.")
        return
    else:
        model_detect_emotions('cnn')


def feedforward_button_click():
    if found_faces is None:
        print("No image selected.")
        return
    else:
        model_detect_emotions('feedforward')


def transferlearning_button_click():
    if found_faces is None:
        print("No image selected.")
        return
    else:
        model_detect_emotions('transferlearning')


def model_detect_emotions(model_name):
    face_image = selected_image.copy()
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'cnn':
        model = CNN().to(device)
        model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
    elif model_name == 'feedforward':
        model = NeuralNet(2304, 7).to(device)
        model.load_state_dict(torch.load('./feed-Forward.pth', map_location=torch.device('cpu')))


        model.eval()
        with torch.no_grad():

            img = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img.reshape(1, -1)).float()
            img = img.to(device)
            outputs = model(img)
            _, predictions = torch.max(outputs, 1)
            emotion_label = emotions[predictions]
            print(outputs)
            print(emotion_label)
            return
    elif model_name == 'transferlearning':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_classes = 7
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        state_dict = torch.load('./resnet-transfer.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model = model.to(device)
    model.eval()

    for i, face in enumerate(found_faces):
        with torch.no_grad():
            output = model(face)
            probabilities = torch.softmax(output, dim=1)
            emotion_index = torch.argmax(probabilities).item()
            emotion_label = emotions[emotion_index]
        x, y = coords[i][0], coords[i][1]
        cv2.putText(face_image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    emotions_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    emotions_image = Image.fromarray(emotions_image)
    emotions_image_tk = ImageTk.PhotoImage(emotions_image)
    display_image_label.configure(image=emotions_image_tk)
    display_image_label.image = emotions_image_tk


def contains_eye(face, eyes):
    face_x, face_y, face_w, face_h = face
    for (x, y, w, h) in eyes:
        if x > face_x and x + w < face_x + face_w and y > face_y and y + h < face_y + face_h:
            return True
    return False


def detect_face():
    global selected_image
    selected_image = original_image.copy()
    found_faces.clear()
    coords.clear()
    eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # convert chosen image to grayscale
    gray = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)

    # eyes and faces detection
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # mark the eyes on the image
    for (x, y, w, h) in eyes:
        cv2.rectangle(selected_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # mark the faces on the image
    for face in faces:
        x, y, w, h = face
        #accept_face = contains_eye(face, eyes)
        accept_face = True
        if accept_face:
            # extract the region of interest (face) from the image
            face_roi = gray[y:y + h, x:x + w]
            # resize the image
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.reshape(1, 1, 48, 48)
            face_roi = face_roi.astype("float32") / 255.0
            # convert face_roi to a PyTorch tensor
            face_tensor = torch.from_numpy(face_roi)
            face_tensor = face_tensor.to(device)
            found_faces.append(face_tensor)
            coords.append((x, y))
        # draw the rectangle that contains a face
        cv2.rectangle(selected_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    haar_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)
    haar_image = Image.fromarray(haar_image)
    haar_image_tk = ImageTk.PhotoImage(haar_image)
    display_image_label.configure(image=haar_image_tk)
    display_image_label.image = haar_image_tk


def choose_image():
    global selected_image, original_image
    found_faces.clear()
    coords.clear()
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # load the chosen image and display it on the window
    image = cv2.imread(file_path)
    selected_image = image.copy()
    original_image = image.copy()
    chosen_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chosen_image = Image.fromarray(chosen_image)
    chosen_image_tk = ImageTk.PhotoImage(chosen_image)
    display_image_label.configure(image=chosen_image_tk)
    display_image_label.image = chosen_image_tk

    # update the window geometry based on the image size
    window_width = chosen_image.width + 20
    window_height = chosen_image.height + 20
    if window_width < 200:
        window_width = 200
    if window_height < 200:
        window_height = 200
    window.geometry(f"{window_width}x{window_height}")


window = tk.Tk()
window.title("Emotion Detection")

window.geometry("960x540")
window.resizable(True, True)

buttons_frame = tk.Frame(window)
buttons_frame.pack(side="right", padx=10)

file_button = tk.Button(buttons_frame, text="Choose Image", command=choose_image)
file_button.pack(pady=10)

face_detection_button = tk.Button(buttons_frame, text="Face Detection", command=detect_face)
face_detection_button.pack(pady=10)

cnn_button = tk.Button(buttons_frame, text="CNN", command=cnn_button_click)
cnn_button.pack(pady=10)

feedforward_button = tk.Button(buttons_frame, text="Feed Forward", command=feedforward_button_click)
feedforward_button.pack(pady=10)

transferlearning_button = tk.Button(buttons_frame, text="Transfer Learning", command=transferlearning_button_click)
transferlearning_button.pack(pady=10)

labels_frame = tk.Frame(window)
labels_frame.pack()

display_image_label = tk.Label(window)
display_image_label.pack(padx=10, pady=10)

window.mainloop()