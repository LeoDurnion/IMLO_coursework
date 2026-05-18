# IMLO-COURSEWORK using Oxford-IIIT Pet Classification

The pet classifcation implements a CNN utilising PyTorch to classify 37 different breeds of cats and dogs via the Oxford-IIIT Pet Classification dataset, where the images were resized to 224x224 and then normalised.

The CNN was trained from scratch, implementing features such as convolutional layers, normalisation and batchnorm

The CNN contains 4 convolutional blocks, each including a convolutional layer, batch normalisation, ReLU and pooling. This is followed by a global pooling to reduce the feature map to size 1x1, and then a flattening function to be passed through the final connected layers.

Adam Optimiser was used at a rate of lr = 0.001 (standard and best after iterations of testing), as well as the standard CrossEntropy.

30 epochs were used to ensure maximal training within the constraint, with validation and evaluation after each epoch. The model that achieved the highest validation accuracy in the end was saved as 'optimal_model.pth' to be used in the 'test.py' file

# Running the program

This program is ran through powershell, so first install the dependencies using:

pip install torch torchvision

Also ensure the dataset is downloaded locally. If not, temporarily change the variable 'download' in the dataloader to True

Run training with:

python training.py

This will display each epoch iteration, alongside the loss and validation accuracy

Run testing with:

python test.py

Which will display the final accuracy percentage
