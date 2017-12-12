import os
import errno
from cv2 import *
from os.path import expanduser, join

# RELATIVE PATH TO THE DIRECTORY
fileDir = os.path.dirname(os.path.realpath('__file__'))
# NEW SECRET FOLDER
filename = os.path.join(fileDir + '/New_secret_pic')


def start_cam():
    # INDEX 0 FOR WEBCAM AND 1 FOR USB WEBCAM.
    cam = cv2.VideoCapture(0)
    # WINDOW_AUTOSIZE (the window size is automatically adjusted to fit the displayed image)
    cv2.namedWindow("test", WINDOW_AUTOSIZE)
    # FUNCTION TO TAKE A SCREEN_SHOT
    take_screen_shot(cam, cv2)
    # RELEASE IMAGE
    destroy_cam(cam, cv2)


def take_screen_shot(cam, cv2):
    # WHILE LOOP TO KEEP RUNNING THE FRAMES
    while True:
        ret, frame = cam.read()
        # RESIZE THE IMAGE FRAME
        resized_image = cv2.resize(frame, (160, 90)) #resize the image frame
        #DISPLAY THE IMAGE
        cv2.imshow("TAKE A SCREEN SHOT", frame)
        # IF NOTHING BREAK THE LOOP
        if not ret:
            break
        # INPUT KEY
        k = cv2.waitKey(1)
        # ESC PRESSED
        if k % 256 == 27:
            # IF ESC BREAK THE LOOP
            print "bye! "
            break
        # SPACE PRESSED
        elif k % 256 == 32:
            # IMAGE NAME
            img_name = "picture.png"
            # WRITE IMAGE TO RELATIVE PATH
            cv2.imwrite(os.path.join(filename, img_name), resized_image)


def destroy_cam(cam, cv2):
    cam.release()
    cv2.destroyAllWindows()


def main():
    start_cam()


main()