import cv2 as cv
import numpy as np

class Frame():

    def __init__(self, frame) -> None:
        self.frame = frame

    def applyMaskToFrame(self, mask):
        '''
        Apply a mask to the frame, so it can visually be seen. No impact on calculations.
        '''
        return cv.bitwise_and(self.frame, self.frame, mask=mask)

    def colorMaskOfFrame(self, upperLeft, lowerRight, lower=[0,0,0], upper=[255,255,255]):
        '''
        Creates a mask based on input of size and color threshold. Default will show all
        colors (do nothing).
        '''
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv.inRange(self.frame, lower, upper)
        return cv.rectangle(mask, upperLeft, lowerRight, 255, -1)


def main(video):
    fishVideo = cv.VideoCapture(video)
    if not fishVideo.isOpened():
        print("Frame didn't load or something")

    while fishVideo.isOpened():
        ret, frame = fishVideo.read() # ret is stupid

        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)

        frameMan = Frame(frame) 

        # Sizes of the two rectangles. Basically split the screen hotdog style.
        upperLeft1, upperLeft2, lowerRight1, lowerRight2 = (0,540), (0,0), (1920,1080), (1920,540)

        # Values to detect red. Which the fish in this are.
        if video == 'videos/fishVideo.mp4':
            lower, upper = [100,50,100], [255,255,255]
        elif video == 'videos/Betta.mp4':
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.GaussianBlur(frame, (21, 21), 0)

        cv.imshow('Blurred', frame)


        if cv.waitKey(1) == ord('q'):
            break
    
    cv.destroyAllWindows()

main('videos/Betta.mp4')