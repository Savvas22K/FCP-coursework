To run our code the following packages are needed: numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, math, random, argparse.

The Ising model does not work with some of the other tasks on some laptops so run them separately if it doesn't initially . The Defuant model saves a graph on the same folder as the python file. When a graph is displayed, to curry on running the model the window showing the figure needs to be closed. To test the networks type in terminal -test_networks

Run the " .py" file in terminal

Git link: https://github.com/Savvas22K/FCP-coursework.git 

For task 1, type -ising_model to call the program, -test_ising to test the model, -alpha N where N is a chosen alpha value and -external M where M is chosen external value.
I was responsible for task 3

For task 2, 
In order to run the model with default parameters：
python <file name>.py -defuant
To run the code with custom beta and threshold values：
python <file name>.py -defuant -beta <beta value> -threshold <threshold value>
To run the test code of defuant model：
python <file name>.py -test_defuant

For task 3 (done by Savvas Kalisperides), type -network N, where N is the number of nodes desired.

For task 4, call the program in the terminal using -ring_network 10, -small_world 10, -small_world 10 -re_wire 10. The flag -re_wire changes the re-wiring probability to the chosen number, you can also change the size of the ring and small world networks by altering the numbers beside the flags in the terminal. 

For task 5 we used the Ising model. Type -ising_model and -use_network N, where N is the number of nodes desired.
