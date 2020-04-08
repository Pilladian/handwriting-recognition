# Python 3.8

# Functionality
#   run python create_tdata.py
#   press space once
#   terminal tells you, which character you should draw
#   draw the character and press space again
#   to exit the program just press esc

import cv2
import numpy
import random

# Path where to save the images
PATH = "NeuralNetwork/data/training/"

# boolean which is true, if left mouse button is pressed
drawing = False
# image file, which will be recognized
input_image = []
shown_image = []

# image variables
size = 500
thickness = 1
amount_squares = 20


# mouse callback function
def draw(event, xCoordinate, yCoordinate, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(shown_image, (xCoordinate, yCoordinate), 20, (0, 0, 0), -1)
        cv2.circle(input_image, (xCoordinate, yCoordinate), 20, (0, 0, 0), -1)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(shown_image, (xCoordinate, yCoordinate), 20, (0, 0, 0), -1)
            cv2.circle(input_image, (xCoordinate, yCoordinate), 20, (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# create input list for hawire
def create_image(character, counter):
    # create image
    mul = size / amount_squares
    for a in range(1, amount_squares + 1):
        xMin = int(a * mul - mul)
        xMax = int(a * mul)

        for b in range(1, amount_squares + 1):
            yMin = int(b * mul - mul)
            yMax = int(b * mul)

            black = False
            for x in range(xMin, xMax):
                for y in range(yMin, yMax):
                    if input_image[x][y][0] == 0:
                        black = True
                        break

            if black:
                for x in range(xMin, xMax):
                    for y in range(yMin, yMax):
                        input_image[x][y] = [0, 0, 0]

    cv2.imwrite(PATH + character + "_training_image_" + counter + ".png", input_image)


# create blank_image
def create_grid():
    global shown_image, input_image

    # empty image
    input_image = 255 * numpy.ones(shape=[size, size, 3], dtype=numpy.uint8)
    shown_image = 255 * numpy.ones(shape=[size, size, 3], dtype=numpy.uint8)
    # create 15x15 grid on image
    for x in range(int(size / amount_squares), size, int(size / amount_squares)):
        cv2.line(shown_image, (x, 0), (x, size), (0, 0, 0), thickness)
    for y in range(int(size / amount_squares), size, int(size / amount_squares)):
        cv2.line(shown_image, (0, y), (size, y), (0, 0, 0), thickness)


# initialize blank image and mouse listener
create_grid()
cv2.namedWindow("HaWiRe - Neural Network")
cv2.setMouseCallback("HaWiRe - Neural Network", draw)

# count data
# digits  0   1   2   3   4   5   6   7   8   9
count = [29, 25, 24, 20, 28, 21, 21, 24, 26, 24]

char = chr(random.randrange(48, 58))
print(" [+] Next character: " + char)
while True:
    cv2.imshow("HaWiRe - Neural Network", shown_image)
    key = cv2.waitKey(20) & 0xFF
    if key == ord(' '):
        create_image(char, str(count[int(char)]))
        count[int(char)] += 1
        create_grid()
        char = chr(random.randrange(48, 58))
        print(" [+] Next character: " + char)
    elif key == 27:
        break

cv2.destroyAllWindows()
