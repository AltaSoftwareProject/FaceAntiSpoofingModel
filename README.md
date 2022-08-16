# FaceAntiSpoofingModel

This Project implements the DeePixBiS model using Python OpenCV, and the Pytorch Framework. This project is inspired from https://github.com/voqtuyen/deep-pix-bis-pad.pytorch and https://github.com/Saiyam26/Face-Anti-Spoofing-using-DeePixBiS.git

### Requirements

- Python 3.6+
- OpenCV
- Numpy
- PyTorch

### Training the Model
1. Run `python Train.py`
2. After Training is complete the program will generate the file "./MyModel.pth", containing weights of the model

### Recognizing
1. Run `python Test.py`
