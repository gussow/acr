This directory includes the following.

1) acr.py - Python file with the code to build and run the model.

2) sample_data_training.txt - Sample data to train the model with.

3) sample_data_predict.txt - Sample data to run the model on.

4) expected.txt - Expected output.

To demo the software, run:
python acr.py

This will train a model on sample_data_training.txt and then predict Acr scores on sample_data_predict.txt. Expected runtime is 5-10 seconds.

This software was tested on python 3.6.7, sklearn 0.21.2 and pandas 0.24.2.

