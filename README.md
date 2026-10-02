SmileyChem

SmileyChem is a Seq2Seq AI model using PyTorch that predicts the result of chemical reactions based on their reactants in SMILES format.

This project's goal was to see if a simple DeepRNN using an Encoder-Decoder approach could predict chemical reactions accurately.




Install dependencies:

pip install -r requirements.txt




How to use:

The file model.py contains the classes DeepRNN, Encoder, Decoder and Seq2Seq and is the core of the model.

The file train.py does the training and saves the model in model.pth or model_v01.pth or model_v02.pth depending on the version.

The file database.txt is a small dataset of simple chemical reactions in SMILES format.

The files are in the corresponding folder v00, v01..

In the version v02 a file tokeniser.py is added to improve tokenisation

In the version v03 an improved tokeniser.py, a canonicalisation.py for canonicalising smiles, 
a validation.py for validating and evaluating sequences, a parser.py to parse reactions, a database.py
to load, store and modify reaction databases and a run.py file to load and test the model 

The v03 also now uses bucket_batching and is TPU-friendly with masking implemented. 

In order to use the saved model you just load it using torch (works until model v02):
  model = torch.load("model.pth", weights_only=False)

In order to use the saved model (for v03) you just load it using torch however you have specifically
load the model as there are many utils saved in the model.pth file:

    model = md.Seq2Seq(vocab_size+3, 256, 3, 256, vocab_size, vocab_size+1, vocab_size + 2)
    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"])

Technologies used:

This Python-based project only uses PyTorch, rdkit and some built-in Python libraries.




The goal of this project was only educational, and therefore this project is open-source. It currently does not have a specified license.
