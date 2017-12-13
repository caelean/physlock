import os
import errno
from cv2 import *
import tkMessageBox
import numpy
from rpc_client import PredictClient

# RELATIVE PATH TO THE DIRECTORY
fileDir = os.path.dirname(os.path.realpath('__file__'))
# NEW SECRET FOLDER
filename = os.path.join(fileDir + '/New_secret_pic')


def start_cam():
    # INDEX 0 FOR WEBCAM AND 1 FOR USB WEBCAM.
    cam = cv2.VideoCapture(0)
    # FUNCTION TO TAKE A SCREEN_SHOT
    answer = take_screen_shot(cam, cv2)
    if answer > 0.5:
        pop_window("CONGRATS," "THE LOCK HAS BEEN UNLOCKED")
    elif answer <= 0.5:
        pop_window("SORRY", "THAT IS NOT THE KEY")
    # RELEASE IMAGE
    destroy_cam(cam, cv2)


# FUNCTION TO DISPLAY THE MESSAGE
def pop_window(title, message):
    tkMessageBox.showinfo(title, message)


# FUNCTION TO TAKE A SCREENSHOT
def take_screen_shot(cam, cv2):
    # WHILE LOOP TO KEEP RUNNING THE FRAMES
    while True:
        ret, frame = cam.read()
        # RESIZE THE IMAGE FRAME
        resized_image = cv2.resize(frame, (160, 90)) #resize the image frame
        # WINDOW NAME
        window_name = "TAKE A SCREEN SHOT"
        # DISPLAY THE IMAGE
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
            # WRITE/SAVE IMAGE TO RELATIVE PATH
            cv2.imwrite(os.path.join(filename, img_name), resized_image)
            # FOR DEBUGGING
            #print resized_image
            # DEFAULT IMAGES ARE (BGR) FORMAT IN OPENCV SO WE NEED TO CHANGE THE ORDER TO RGB BEFORE SENDING IT.
            b, g, r = cv2.split(resized_image)
            print "blue"
            print b
            print "green"
            print g
            print "red"
            print r
            # MERGE NEW ORDER (RGB)
            resized_image = cv2.merge((r, g, b))
            print "old"
            print resized_image
            # TRANSFORM THE
            new_resized_image = numpy.array(resized_image)
            # COULD BE USING .T.
            flattened_image = new_resized_image.flatten()
            print "new"
            print flattened_image
            client = PredictClient('127.0.0.1', 9000, 'physlock', 1513186008)
            answer = client.predict(flattened_image)['prediction'].float_value
            return answer

# FOR DEBUGGING
#print "new one"
#print resized_image


# STOP CAMERA AND REMOVE WINDOWS
def destroy_cam(cam, cv2):
    cam.release()
    cv2.destroyAllWindows()


def main():
    pop_window("INSTRUCTIONS", "PRESS SPACE TO VERIFY AND ESC TO EXIT THE PROGRAM")
    start_cam()


main()
