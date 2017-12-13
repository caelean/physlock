import os
import errno
from cv2 import *
import tkMessageBox

# RELATIVE PATH TO THE DIRECTORY
fileDir = os.path.dirname(os.path.realpath('__file__'))
# NEW SECRET FOLDER
filename = os.path.join(fileDir + '/New_secret_pic')


def start_cam():
    # INDEX 0 FOR WEBCAM AND 1 FOR USB WEBCAM.
    cam = cv2.VideoCapture(0)
    # FUNCTION TO TAKE A SCREEN_SHOT
    take_screen_shot(cam, cv2)
    # RELEASE IMAGE
    destroy_cam(cam, cv2)


def pop_window(title, message):
    tkMessageBox.showinfo(title, message)


def take_screen_shot(cam, cv2):
    # WHILE LOOP TO KEEP RUNNING THE FRAMES
    while True:
        ret, frame = cam.read()
        # RESIZE THE IMAGE FRAME
        resized_image = cv2.resize(frame, (160, 90)) #resize the image frame
        #DISPLAY THE IMAGE
        window_name = "TAKE A SCREEN SHOT"
        cv2.imshow(window_name, frame)
        # MOVE THE WINDOW TO THE CENTER
        cv2.moveWindow(window_name, 100, 0)
        # IF NOTHING BREAK THE LOOP
        if not ret:
            break
        # WAIT FOR INPUT KEY
        k = cv2.waitKey(1)
        # ESC PRESSED
        if k % 256 == 27:
            # IF ESC BREAK THE LOOP
            print "bye! "
            destroy_cam(cam, cv2)
            exit()
            break
        # SPACE PRESSED
        elif k % 256 == 32:
            # IMAGE NAME
            img_name = "picture.jpg"
            # WRITE IMAGE TO RELATIVE PATH
            cv2.imwrite(os.path.join(filename, img_name), resized_image)


def destroy_cam(cam, cv2):
    cam.release()
    cv2.destroyAllWindows()


def main():
    pop_window("INSTRUCTIONS", "PRESS SPACE TO VERIFY AND ESC TO EXIT THE PROGRAM")
    start_cam()


main()
