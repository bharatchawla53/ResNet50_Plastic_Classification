# Bharat Chawla and Himaja R. Ginkala
# This class {TODO} for recognizing plastics into heavy plastic, no plastic, some plastic, and no plastic. 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import statements
import cv2
import sys
import torch
from PIL import Image

# main function - TODO: add description
def main(argv):

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Images
    for f in 'zidane.jpg', 'bus.jpg':

        torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
    im1 = Image.open('zidane.jpg')  # PIL image
    im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    # Inference
    results = model([im1, im2], size=640) # batch of images
    results.print()

    results.save()

    return
if __name__ == "__main__":

    main(sys.argv)
