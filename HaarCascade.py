import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

class HaarCascade:
    def __init__(self, path):
        self.path = path
        self.face_cascade = cv2.CascadeClassifier(path)

    def detect(self, image_tensor):
        image_np = np.array(image_tensor)
        # gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray_image = image_np
        cv2.imshow('Detected Faces', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        return image_np, faces
