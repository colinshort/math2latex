Project Structure:
    math2latex/src/ (source code):
       - CNN.py is the implementation of the neural network
       - data_process.py reads the data from my local machine and packages it into data.pkl and labels.csv
       - math2latex.py is the main program that implements the input data processing and classification
Each .py file in math2latex/src/ is individually executable. However, the CNN.py depends on the output from data_process.py and math2latex.py depends on the output from both data_process.py and CNN.py.
    The full program can be executed by navigating to math2latex/src/ and running 'python data_process.py' into the command line.
    Next, run 'python CNN.py'. Finally, run 'python math2latex.py'. The output will be given in the command line.

    The inputs to math2latex.py are images only and will be drawn from the math2latex/inputs/ directory.
    This project was only been tested in a Linux command line environment, so it should be run from a Linux command line. 
    Additionally, here are all of the dependencies:
        - OpenCV
        - NumPy
        - pandas
        - TensorFlow
        - TensorFlow Keras
        - Matplotlib
        - scikit-learn

References
[1] Memon, J., Sami, M., Khan, R. A., and Uddin, M. Handwritten Optical Character Recognition (OCR): A Comprehensive Systematic Literature Review (SLR). IEEE Access 8 (2020), 142642–142668.
[2] Mishra, A., Ram, A. S., and C, K. Handwritten Text Recognition Using Convolutional Neural Network.
[3] Mouch`ere, H. CHROME: Competition on Recognition of Online Handwritten Mathematical Expressions, Feb 2014.
[4] Nano, X. Handwritten Math Symbols Dataset, Jan 2017.
[5] Rosebrock, A. OCR: Handwriting recognition with OpenCV, Keras, and TensorFlow, Jun 2023.
[6] Yuan, Y., Liu, X., Dikubab, W., Liu, H., Ji, Z., Wu, Z., and Bai, X. Syntax-Aware Network for Handwritten Mathematical Expression Recognition, 2022.
[7] Zhang, T. Y., and Suen, C. Y. A Fast Parallel Algorithm for Thinning Digital Patterns. Communications of the ACM 27, 3 (1984), 236–239.