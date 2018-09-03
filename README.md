# Handwritten Digit Recognition with a Multilayer Perceptron
Multi-Layer Perceptron tailored towards the MNIST data set.

<img src="https://media.data.world/1ECoxJH9QVCIn3km6L5e_Screen%20Shot%202017-06-27%20at%204.58.43%20PM.png" width=400/>

This project was my first time coding a machine learning algorithm. My initial approach to coding the MLP was poor and turned into a mess. I struggled with a chain reaction of misaligned matrixes and overflows from small values being calculated. However, everyone has to start somewhere right? The primary data structure I used was NumPy's n-dimensional arrays for fast vector operations.

### MLP Architecture

Input size is 784 pixels and the output is 10 digits (0-9). I used a single hidden layer with twenty nodes inside based on a Stats StackExchange answer with this handy formula:

<img src="http://i67.tinypic.com/emz51.jpg" width=200/> <img src="http://i67.tinypic.com/2qsmo28.jpg" width=200/>

[Click here to see the original post.](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)

I set my alpha to two (two degrees of freedom) and ended up with a value of 18.9 which I simply rounded up to twenty. I never strayed far from that amount of hidden neurons.

### Classifier Effectiveness

For measuring the effectiveness, I used a confusion matrix. The total accuracy for my finished model was 90.7%.

<img src="http://i63.tinypic.com/5kewe8.png" width=400/>

### Model Training Length

It took thirty to sixty seconds to read in the MNIST dataset and pre-process it. After that, it ran about two minutes max rotating through a total of ten epochs, each pushing 30,000 individual pieces of data through the network. 

### Choice of Training
I went with sequential training instead of batch for a few different reasons. First, it was easy to program. Second, it seemed intuitive that the MLP would train much faster in this fashion since it updates its weights with each piece of data being sent through the network. As I found out, this was true since my model during its first epoch was already at a total accuracy of 88.5%. It only increased from there. If I didn’t get such a healthy response, I would have toyed with the idea of batch or mini-batch training but that didn’t happen this time. Another reason why I chose sequential training is because I personally enjoyed seeing the progress of the ML algorithm with each piece of data.

<img src="http://i64.tinypic.com/ridb8g.png" width=400/> <img src="http://i66.tinypic.com/2pu03g4.png" width=400/>

### Analysis of the MLP

Overall I was very surprised with how this project turned out. I ran into plenty of road blocks but when I finally had a working program, it was very efficient. The training took only a few minutes with an overall 90.7% accuracy, the validation data had a 93.14% accuracy, and the test data had a 93.0% accuracy. 

<img src="http://i64.tinypic.com/2houv4n.png" width=400/>

The numbers my model had the most trouble with were threes and eights which read them as twos. The only conclusion I can come up with is that the numbers have lighter grey pixels and it is possible that the model considers the values to be more important than they should. My current model would not be the best for predicting those values as accurately as others. 
