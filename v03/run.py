import torch
import torch.nn as nn
import model_v03 as md
import tokeniser as t
import parser
import validation
import database as db
import pickle
from pathlib import Path




CSV_RAW_REACTIONS_FILE_NAME = "raw_test.csv"
PICKELED_REACTION_DATABASE_FILE_NAME = "reactions_test.db"
PTH_MODEL_NAME = "model.pth"




reactions_file = Path(__file__).parent / PICKELED_REACTION_DATABASE_FILE_NAME
model_file = Path(__file__).parent / PTH_MODEL_NAME

if reactions_file.is_file():

  with open(PICKELED_REACTION_DATABASE_FILE_NAME, "rb") as file:

    database = pickle.load(file)

else:

  database = db.Database(CSV_RAW_REACTIONS_FILE_NAME, db.ignore)
  database.load_data()
  database.canonicalise_mapped()
  database.tokenise()

  with open(PICKELED_REACTION_DATABASE_FILE_NAME, "wb") as file:

    pickle.dump(database ,file)



vocab_size = t.SmilesReader.vocab_size()

model = md.Seq2Seq(vocab_size+2, 256, 3, 256, vocab_size, vocab_size+1)

if model_file.is_file():
  try:
    state_dict = torch.load("model.pth", weights_only=True)
    model.load_state_dict(state_dict)

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  raise ValueError(f"This file does not exist: {PTH_MODEL_NAME}")

criteron = nn.CrossEntropyLoss(ignore_index=-100)

examine_loss = 0

for u in range(len(database.reactions)):
  src = database.reactions[u].reactants_tokens
  trg = torch.cat([torch.tensor([vocab_size]),database.reactions[u].products_tokens, torch.tensor([vocab_size+1])])

  logits = model.forward(src)

  if logits.shape[0] < trg[1:].shape[0]:
    pad = torch.full((trg[1:].shape[0] - logits.shape[0],), -100, dtype=trg.dtype, device=trg.device)
    new_target = torch.cat([trg[1:], pad])
  else:
    new_target = trg[1:logits.shape[0]]

  loss = criteron(logits, new_target)

  examine_loss += loss.item()



print(examine_loss)
