import cv2
import threading as th

"""
This class was written to be used as a threading implementation for the main AmplifAI project program; however,
it did not work as intended and has since been deprecated for now.

@author Ethan Smith
@date 11.10.2024
"""
class VideoCaptureAsync:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.ret, self.frame = self.cap.read()
        self.running = False
        self.thread = None

    """
    This function 'starts' the async operations used in the main AmplifAI project program.
    
    @param self: VideoCaptureAsync
    @returns None
    """
    def start(self):
        if self.running:
            RuntimeWarning("Video Capture Already Running")
            return
        self.running = True
        self.thread = th.Thread(target=self.update, args=())
        self.thread.start()

    """
    This function is basically a loop to update each frame as it is called
    
    @param self: VideoCaptureAsync
    @returns None
    """
    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    """
    This function simply returns the current boolean value of self.ret and the current
    state of the Union type frame
    
    @param self: VideoCaptureAsync
    @returns self.ret: boolean, self.frame: Union[Mat, ndarray[Any, dtype], ndarray]
    """
    def read(self):
        return self.ret, self.frame

    """
    This function stops the threading used by this program when the stop method is called
    
    @param self: VideoCaptureAsync
    @returns None
    """
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.cap.release()

    """
    Simply the deconstructor for the class
    
    @param self: VideoCaptureAsync
    @returns None
    """
    def __del__(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
