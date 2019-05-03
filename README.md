
# A Deep Learning-based Method for the Design of Microstructural Materials

The test dataset for the model can be downloaded from the link: https://1drv.ms/f/s!Ai_nIBFtgsTMgTrizTMZYov-7VD2

The test datasets are suitable to used for:
(i)  Convolutional Neural Network for Compliance Prediction
(ii) Inverse Design Model


## Design Candidate Generator
The Deep Convolutional Generative Adversarial Network can be trained by providing the data for training and running main.py in the DCGAN folder.
To generate new images based on the pretrained checkpoint, make sure that the checkpoint folder is located in the same directory.
The model.py file should also be modified accordingly.

## Design Evaluator
The Convolutional Neural Network for Compliance Tensor Prediction can be demonstrated on the test dataset by running the cnn_compliance.py file. For different dataset (eg: Ellipse or Circle & Square), the code in the file should be modified according to comment in the code.

## Design Network
The combined inverse design network can be demonstrated by running the main.py file in the inv_analysis folder. The checkpoint folder should also be placed in the same directory. For the demonstration for different datasets (eg: Ellipse or Circle & Square), the code should be modified according to comment in the code.
