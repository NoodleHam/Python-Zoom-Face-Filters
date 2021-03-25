''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
from collections import OrderedDict

import cv2
import os
import urllib.request as urlreq
import numpy as np
from helpers import visualize_facial_landmarks, get_biggest_face, get_facial_landmark_dict, get_image_dict_from_hulls, paste_alpha_img, eye_to_cartoon_eye, visualize_eye_img

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

landmarks_dict = get_facial_landmark_dict()
# landmarks_filter_locs_dict = {"nose": [305, 387], "left_eye": [187, 370]}
landmarks_filter_locs_dict = OrderedDict([
    ("mouth", (305, 468)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (187, 370)),
    ("left_eye", (424, 370)),
    ("nose", (305, 387)),
    ("jaw", (0, 17))
])

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    if (haarcascade in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

filter_img = cv2.imread("../filter_imgs/cat1.jpg")
eye_bg_img = cv2.imread("../eye_imgs/eye_background_2.png", cv2.IMREAD_UNCHANGED)
eye_bg_img = cv2.resize(eye_bg_img, (96,96))
left_eyeball = cv2.imread("../eye_imgs/left_eyeball.png", cv2.IMREAD_UNCHANGED)
left_eyeball = cv2.resize(left_eyeball, (96,96))
right_eyeball = cv2.imread("../eye_imgs/right_eyeball.png", cv2.IMREAD_UNCHANGED)
right_eyeball = cv2.resize(right_eyeball, (96,96))
assert filter_img is not None
# get image from webcam
print("checking webcam for connection ...")
webcam_cap = cv2.VideoCapture(0)

while (True):
    # read webcam
    _, frame = webcam_cap.read()
    frame = cv2.flip(frame, 1)
    visualization = None
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    # this returns [face1, face2, ...] where face_n is a length-4 array representing a bounding box
    faces = detector.detectMultiScale(gray)
    if len(faces) > 0:
        # only consider the biggest face detected
        faces = get_biggest_face(faces)

        # Detect landmarks on "gray"
        # this returns [landmark_1, landmark_2, ...] where landmark_n is a 1x68x2 array representing the locations
        # of the 68 facial landmarks
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        visualization, hull_dict = visualize_facial_landmarks(frame, landmarks[0][0])
        # print(hull_dict['nose'])
        face_regions = get_image_dict_from_hulls(frame, hull_dict)
        final_img = filter_img.copy()
        for name in ['mouth', 'left_eye', 'right_eye']:
            if name == 'left_eye':
                visualize_eye_img(face_regions[name])

        # cv2.imshow('filtered', final_img)
        # for landmark in landmarks:
        #     for x, y in landmark[0]:
        #         # display landmarks on "frame/image,"
        #         # with blue colour in BGR and thickness 2
        #         cv2.circle(frame, (x, y), 1, (255, 0, 0), 2)

    # save last instance of detected image
    # cv2.imwrite('face-detect.jpg', frame)

    # Show image
    if visualization is not None:
        cv2.imshow("visualization", visualization)
    else:
        cv2.imshow("visualization", frame)

    # terminate the capture window
    if cv2.waitKey(20) & 0xFF == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break
