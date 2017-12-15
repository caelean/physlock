import os
import errno
from cv2 import *
import tkMessageBox
import numpy
import matlab
import matlab.engine as engine
from rpc_client import PredictClient

CHOSEN_KEY = 2
THRESHOLD = 5.5

# RELATIVE PATH TO THE DIRECTORY
fileDir = os.path.dirname(os.path.realpath('__file__'))
# NEW SECRET FOLDER
filename = os.path.join(fileDir)

unlocked = False


def start_cam():
    # INDEX 0 FOR WEBCAM AND 1 FOR USB WEBCAM.
    cam = cv2.VideoCapture(1)
    # FUNCTION TO TAKE A SCREEN_SHOT
    while True:
        answer = take_screen_shot(cam, cv2)
        print answer
        if answer == CHOSEN_KEY:
            pop_window("CONGRATS", "THE LOCK HAS BEEN UNLOCKED")
            unlocked = True
        else:
            pop_window("SORRY", "THAT IS NOT THE KEY")
            unlocked = False
    # RELEASE IMAGE
    destroy_cam(cam, cv2)


# FUNCTION TO DISPLAY THE MESSAGE
def pop_window(title, message):
    tkMessageBox.showinfo(title, message)


# FUNCTION TO READ MEAN AND STANDARD DEVIATION OF OUR DATA
def read_mean_std():
    with open('mean_std.txt') as f:
        lines = f.readlines()
        means = lines[0].split(',')
        stds = lines[1].split(',')
    return means, stds

# FUNCTION TO NORMALIZE DATA TO STANDARD DEVIATIONS FROM MEAN
def normalize(data):
    means, stds = read_mean_std()
    for i in range(3):
        data[i] = (data[i] - float(means[i]))/float(stds[i])
    return data

# FUNCTION TO REMOVE DATA PAST A THRESHOLD OF
# STANDARD DEVIATIONS FROM MEAN (SIGNIFICANT OUTLIER)
def screen(data):
    return sum([abs(i) for i in data]) <= THRESHOLD

# FUNCTION TO TAKE A SCREENSHOT
def take_screen_shot(cam, cv2):
    # WHILE LOOP TO KEEP RUNNING THE FRAMES
    while True:
        ret, frame = cam.read()
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
            img_name = "test.jpg"
            # WRITE/SAVE IMAGE TO RELATIVE PATH
            cv2.imwrite(os.path.join(filename, img_name), frame)
            # CALL MATLAB IMAGE FEATURE EXTRACT SCRIPT
            eng = engine.start_matlab()
            data = eng.capture_data()[0]
            # USE MEAN NORMALIZATION TO AID NN TRAINING
            data = normalize(data)
            # SCREEN OUT OUTLIER DATA BASED ON STANDARD DEVIATIONS
            if(not screen(data)):
                return -1
            # CONNECT TO MODEL ON LOCAL HOST
            client = PredictClient('127.0.0.1', 9000, 'physlock', 1513238173)
            answer = client.predict([float(i) for i in data])[0]
            return answer


# STOP CAMERA AND REMOVE WINDOWS
def destroy_cam(cam, cv2):
    cam.release()
    cv2.destroyAllWindows()


def main():
    pop_window("INSTRUCTIONS", "PRESS SPACE TO VERIFY AND ESC TO EXIT THE PROGRAM")
    start_cam()


main()
