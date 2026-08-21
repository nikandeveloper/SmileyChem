SmileyChem

SmileyChem is a Seq2Seq AI model using PyTorch that predicts the result of chemical reactions based on their reactants in SMILES format.

This project's goal was to see if a simple DeepRNN using an Encoder-Decoder approach could predict chemical reactions accurately.



Install dependencies:

pip install -r requirements.txt



How to use:

The file model.py contains the classes DeepRNN, Encoder, Decoder and Seq2Seq and is the core of the model.

The file train.py does the training (however it does not save the weights and biases yet).

The file database.txt is a small dataset of simple chemical reactions in SMILES format.



Technologies used:

This Python-based project only uses PyTorch and some built-in Python libraries.



The goal of this project was only educational, and therefore this project is open-source. It currently does not have a specified license.
