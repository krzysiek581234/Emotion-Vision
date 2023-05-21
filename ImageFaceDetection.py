import cv2
import torch

from CNN import CNN


class ImageFaceDetection:
    def __init__(self, face_cascade_path, eyes_cascade_path, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eyes_cascade = cv2.CascadeClassifier(eyes_cascade_path)
        self.model_path = model_path
        self.model = CNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def contains_eye(self, face, eyes):
        face_x, face_y, face_w, face_h = face
        for (x, y, w, h) in eyes:
            if x > face_x and x + w < face_x + face_w and y > face_y and y + h < face_y + face_h:
                return True
        return False

    def detect_faces(self, image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Eyes Detection
        eyes = self.eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # Mark the eyes on the image
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Perform face detection
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Filter faces for based on at least one eye detection
        for face in faces:
            x, y, w, h = face
            accept_face = self.contains_eye(face, eyes)
            if accept_face:
                # Extract the region of interest (face) from the grayscale image
                face_roi = gray[y:y+h, x:x+w]
                # Preprocess the face for emotion classification
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.reshape(1, 1, 48, 48)
                face_roi = face_roi.astype("float32") / 255.0
                # Convert face_roi to a PyTorch tensor
                face_tensor = torch.from_numpy(face_roi)
                face_tensor = face_tensor.to(self.device)
                # Classify face emotion using CNN model
                self.model.eval()
                with torch.no_grad():
                    output = self.model(face_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    emotion_index = torch.argmax(probabilities).item()
                    emotion_label = self.emotion_labels[emotion_index]

                # Draw face rectangle and label
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Display the marked image
        cv2.imshow("Faces", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Usage example
if __name__ == '__main__':
    face_cascade_path = 'haarcascade_frontalface_default.xml'  # Path to the Haar cascade XML file
    eyes_cascade_path = 'haarcascade_eye.xml'
    image_path = './img/happy2.jpg'  # Path to the input JPEG image

    face_detector = ImageFaceDetection(face_cascade_path, eyes_cascade_path, './model/model.pth')
    face_detector.detect_faces(image_path)
