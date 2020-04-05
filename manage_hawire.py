# Python 3.8

from pathlib import Path
from NeuralNetwork import NeuralNetwork
import cv2
import os


# create weights for the neural network
os.system("clear")
wih1_file = Path("NeuralNetwork/data/wih1.npy")
wh1h2_file = Path("NeuralNetwork/data/wh1h2.npy")
wh2o_file = Path("NeuralNetwork/data/wh2o.npy")

decision = "no"
if wih1_file.is_file() or wih1_file.is_file() or wih1_file.is_file():
    print("\n [!] Found already existing weight file(s) : NeuralNetwork/data/...")
    decision = input(" [?] Skip creation of weights [y/N]: ")

if decision in ['y', 'Y', 'yes', 'Yes']:
    print("\n [+] Using already existing weights\n")
else:
    print("\n [+] Creating weights for neural network")
    NeuralNetwork.create_weights()
    print(" [+] Done\n")

# initialize new neural network
print(" [+] Initialize neural network")
neural_network = NeuralNetwork.HaWiRe()
print(" [+] Done\n")

# train neural network
decision = input(" [?] Train your neural network [y/N]: ")
if decision not in ['y', 'Y', 'yes', 'Yes']:
    exit(1)
else:
    os.system("clear")
    waves = input(" [?] How many loops with the same training data (default 20): ")
    if waves == "":
        runs = 20
    else:
        runs = int(waves)
    path_training_data = "./NeuralNetwork/data/training/"
    os.system("clear")
    for times in range(runs):
        for root, dirs, files in os.walk(path_training_data, topdown=False):
            for file in files:
                # get training image
                img = cv2.imread(path_training_data + file)
                # create its input_list
                input_list = []
                mul = 500 / 20
                for a in range(20):
                    xMin = int(a * mul - mul)
                    xMax = int(a * mul)

                    for b in range(20):
                        yMin = int(b * mul - mul)
                        yMax = int(b * mul)

                        black = False
                        for x in range(xMin, xMax):
                            for y in range(yMin, yMax):
                                if img[x][y][0] == 0:
                                    black = True
                                    break

                        if black:
                            input_list.append(0.99)
                        else:
                            input_list.append(0.01)

                # create target_list
                target_list = []
                num = int(file[0])
                for a in range(10):
                    target_list.append(0.01)
                target_list[num] = 0.99

                # train neural network
                neural_network.train(input_list, target_list)

    print("\n [+] Training finished")
    print(" [+] Saving weights")
    neural_network.save_weights()
    print(" [+] Done")
