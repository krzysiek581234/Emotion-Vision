import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from CNN import CNN
from NeuralNet import NeuralNet
import torch

selected_image = None

def cnn_button_click():
    if selected_image is None:
        print("No image selected.")
        return

    cnn(selected_image)

def cnn(image):
    model = CNN()  # Create an instance of the model
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    # Preprocess the image
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    predicted_emotion = emotions[predicted.item()]

    predicted_label.configure(text="Predicted Emotion: " + predicted_emotion)

def detect_face():
    global selected_image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    original_image = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        red_square_area = original_image[y:y+h, x:x+w]
        red_square_area = cv2.resize(red_square_area, (48, 48), interpolation=cv2.INTER_AREA)
        red_square_area = cv2.cvtColor(red_square_area, cv2.COLOR_BGR2GRAY)
        selected_image = red_square_area.copy()

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = Image.fromarray(original_image)
    original_image = original_image.resize((300, 300), Image.Resampling.LANCZOS)
    original_image_tk = ImageTk.PhotoImage(original_image)

    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_image)
    processed_image = processed_image.resize((300, 300), Image.Resampling.LANCZOS)
    processed_image_tk = ImageTk.PhotoImage(processed_image)

    original_label.configure(image=original_image_tk)
    original_label.image = original_image_tk
    processed_label.configure(image=processed_image_tk)
    processed_label.image = processed_image_tk

    red_square_area = cv2.cvtColor(red_square_area, cv2.COLOR_BGR2RGB)
    red_square_area = Image.fromarray(red_square_area)
    red_square_area = red_square_area.resize((300, 300), Image.Resampling.LANCZOS)
    red_square_area_tk = ImageTk.PhotoImage(red_square_area)

    extracted_label.configure(image=red_square_area_tk)
    extracted_label.image = red_square_area_tk

window = tk.Tk()
window.title("Face Detection")

window.geometry("990x540")
window.resizable(False, False)

file_button = tk.Button(window, text="Choose Image", command=detect_face)
file_button.pack(pady=10)

labels_frame = tk.Frame(window)
labels_frame.pack()
                   
text_label1 = tk.Label(labels_frame, text="Original Image", font=("Arial", 12))
text_label1.pack(side="left", padx=100)

text_label2 = tk.Label(labels_frame, text="Processed Image", font=("Arial", 12))
text_label2.pack(side="left", padx=100)

text_label3 = tk.Label(labels_frame, text="Extracted Area", font=("Arial", 12))
text_label3.pack(side="left", padx=100)

placeholder_image = Image.new("RGB", (300, 300), "white")
placeholder_image_tk = ImageTk.PhotoImage(placeholder_image)

pictures_frame = tk.Frame(window)
pictures_frame.pack()

original_label = tk.Label(pictures_frame, image=placeholder_image_tk)
original_label.pack(side="left")

processed_label = tk.Label(pictures_frame, image=placeholder_image_tk)
processed_label.pack(side="left")

extracted_label = tk.Label(pictures_frame, image=placeholder_image_tk)
extracted_label.pack(side="left")

buttons_frame = tk.Frame(window)
buttons_frame.pack(pady=10)

cnn_button = tk.Button(buttons_frame, text="CNN", command=cnn_button_click)
cnn_button.pack(side="left", padx=10)

feed_forward_button = tk.Button(buttons_frame, text="Feed Forward")
feed_forward_button.pack(side="left", padx=10)

predicted_label = tk.Label(window, text="Predicted Emotion: ", font=("Arial", 12))
predicted_label.pack(pady=10)

window.mainloop()
