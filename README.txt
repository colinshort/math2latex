Project Structure:
    math2latex/src/: source code - CNN.py is the implementation of the neural network, data_process.py reads the data from my local machine and packages it into data.pkl and labels.csv, 
                                    math2latex.py is the main program that implements the input data processing and classification
Each .py file in math2latex/src/ is individually executable. However, the CNN.py depends on the output from data_process.py and math2latex.py depends on the output from both data_process.py and CNN.py.
    Since we have the data and model already stored, the only file that should be executed is math2latex.py. This should be done by navigating to math2latex/src/ and entering 'python math2latex.py' into the command line.
    The output will be given in the command line.
    The inputs to math2latex.py are images only, and will be drawn from the math2latex/inputs/ directory.
    This project was only been tested in a Linux command line environment, so it should be run from a Linux command line. 
    Additionally, here are all of the dependcies:
        - OpenCV
        - NumPy
        - pandas
        - TensorFlow
        - TensorFlow Keras
        - Matplotlib
        - scikit-learn