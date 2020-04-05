# Python 3.8

from NeuralNetwork import NeuralNetwork
import numpy
import cv2


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


# create input list for hawire
def create_inputs():
    # contains lists of coordinates of each square
    inp_list = []

    mul = size / amount_squares
    for a in range(amount_squares):
        xMin = int(a * mul - mul)
        xMax = int(a * mul)

        for b in range(amount_squares):
            yMin = int(b * mul - mul)
            yMax = int(b * mul)

            black = False
            for x in range(xMin, xMax):
                for y in range(yMin, yMax):
                    if input_image[x][y][0] == 0:
                        black = True
                        break

            if black:
                inp_list.append(0.99)
            else:
                inp_list.append(0.01)

    return inp_list


# interpret output of hawire
def interpret_output(max_ind):
    return " [+] HaWiRe thinks: " + str(max_ind)


# initialize blank image and mouse listener
create_grid()
cv2.namedWindow("HaWiRe - Neural Network")
cv2.setMouseCallback("HaWiRe - Neural Network", draw)

# create neural network
hawire = NeuralNetwork.HaWiRe()

while True:
    cv2.imshow("HaWiRe - Neural Network", shown_image)
    key = cv2.waitKey(20) & 0xFF

    if key == ord(' '):
        # create input list
        input_list = create_inputs()
        print(interpret_output(hawire.query(input_list)))

        create_grid()
    elif key == 27:
        break
cv2.destroyAllWindows()
