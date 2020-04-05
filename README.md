# hawire
Neural Network Implementation in Python, which can be trained to recognize handwritten letters and digits

Features:
  - the implementation of the neural network does not need to be modified, to be used for other projects 
  - with manage_hawire.py the weights can be created and the network can be trained
  - besides the training data in data/training, new data can be created with create_tdata.py
  - query the network with self drawn letters and digits
  
The library which I used for the drawing is opencv (https://github.com/opencv/opencv).
The implementation of the neural network is based on the knowledge of the book 'Neuronale Netze selbst programmieren'.

Functionality:
  - you can use the already trained network by simply executing hawire.py or you can create your own network and train it by yourself

  - to do so create weights with manage_hawire.py
  - either create your own training data or just use the given examples to train your network
  - query the network by executing hawire.py and drawing the image
  - you need to press space to query the network, see its conclusion and free the space for another drawing
