import cv2
from tkinter import Tk, Label
from PIL import Image, ImageTk
from ultralytics import YOLO
import math
import mediapipe as mp
import PyInstaller.__main__ as pyim

"""
This function is used to track the hands of the user within the 'eyesight' of the camera.

@author Aditee Gautam
@version 05.16.2024
"""
#@see https://gautamaditee.medium.com/hand-recognition-using-opencv-a7b109941c88
class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):

                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            print("Hands Keypoint")
            print(bbox)
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmsList, bbox

    def findFingerUp(self):
        fingers = []

        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):

        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0.255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

"""
This class was made as the main proprietor of the AmplifAI project. It is the class that
everything depends on, especially my grade. It combines a multitude of functions written
over the course of about a month and a half.

@author Ethan Smith
@version 11.20.2024
"""
class GuitarDetectionApp:
    def __init__(self, window, htd_class_obj: HandTrackingDynamic):
        self.detector = htd_class_obj  #init a new instance of HandTrackingDynamic class

        self.window = window
        self.window.title("AmplifAI")

        #window icon
        ico = Image.open('imgs/pixelated-electric-guitar-isolated-vector-18696375-removebg-preview.png')
        photo = ImageTk.PhotoImage(ico)
        self.window.wm_iconphoto(False, photo)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        #convert format to MJPEG FROM YUYU default
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter.fourcc(*'MJPG'))  #increased the frame rate a little bit but not enought to matter

        #load my custom trained YOLO model
        self.model_path = 'model_training/best.pt'
        self.model = YOLO(self.model_path)

        #set up label for video display
        self.label = Label(window)
        self.label.pack()

        #start video loop
        self.update_video()

    """
    This function is called every time 'self.window.after(5, self.update_video)' is called.
    The .after method is basically a looping function that updates the Tkinter window everytime
    it is called.
    
    @param self: GuitarDetectionApp
    @returns None
    """
    def update_video(self):

        #capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            self.__del__()
            return


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model(frame_rgb, stream=True)

        conf_threshold = 0.65

        frame_rgb = cv2.flip(frame_rgb, 1)  # FIXME: only flips when a "guitar" is detected

        #putting the bounding boxes on the frame
        for result in results:
            for box in result.boxes:

                confidence = box.conf[0]

                if confidence < conf_threshold:
                    continue  #skip the low-conf detects

                #grab bounding box coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])  #cast to integers

                #self.model.names holds the class label
                label = f"{self.model.names[int(box.cls[0])]}: {confidence:.2f}"



                #rectangle for bounding box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                #label for confidence
                cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # perform finger detection
        frame_rgb = self.detector.findFingers(frame_rgb)

        #convert the frame back to ImageTk format for Tkinter
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)



        #ipdate label with the new frame
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)



        #repeat after a delay for continuous video feed <- adjust this to alter fps if issues arise
        self.window.after(5, self.update_video)

    def __del__(self):
        #let go of the video capture when the app is closed
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = Tk() #create root object for the window to 'latch' onto
    app = GuitarDetectionApp(root, HandTrackingDynamic())
    root.mainloop() #begin the looping of the program

