import torch
import torch.nn as nn
import model_v03 as md
import tokeniser as t
import parser
import validation
import canonicalisation
import database as db
import pickle
from pathlib import Path




CSV_RAW_REACTIONS_FILE_NAME = "raw_test.csv"
PICKELED_REACTION_DATABASE_FILE_NAME = "reactions_test.db"
PTH_MODEL_NAME = "model.pth"

penalty_importance = 1
ignoring_index = -100


reactions_file = Path(__file__).parent / PICKELED_REACTION_DATABASE_FILE_NAME
model_file = Path(__file__).parent / PTH_MODEL_NAME

if reactions_file.is_file():

  with open(reactions_file, "rb") as file:

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
    state_dict = torch.load("model.pth", weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["model_state_dict"])

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  raise ValueError(f"This file does not exist: {PTH_MODEL_NAME}")

criteron = nn.CrossEntropyLoss(ignore_index=ignoring_index)

examine_loss = 0
validity_rate = 0 
top_1 = 0

for u in range(len(database.reactions)):
  src = database.reactions[u].reactants_tokens.unsqueeze(0)
  trg = torch.cat([torch.tensor([vocab_size]),database.reactions[u].products_tokens, torch.tensor([vocab_size+1])])

  logits = model.forward(src)[0]
  
  d = logits.shape[0] - trg[1:].shape[0]

  factor = d > 0
 
  if factor:
    pad = torch.full((logits.shape[0] - trg[1:].shape[0],), -100, dtype=trg.dtype, device=trg.device)
    padded = torch.cat([trg[1:], pad])
    new_target = padded
  else:
    cut = trg[1:logits.shape[0] + 1]
    new_target = cut

  loss = criteron(logits, new_target)

  loss = loss + (abs(d)+d)/(2*vocab_size)

  smiles = t.SmilesReader.detokenise(logits.argmax(dim=-1))

  valid = True
  separate_logits = smiles.split(".")
  for part in separate_logits:
    if not validation.validate_smiles(part):
      valid = False
  if not valid:
    validity_rate += 0
    print("invalid")
  else:
    validity_rate += 1
    print("valid")
    if canonicalisation.canonicalise_mapped(smiles) == database.reactions[u].products_canonical:
      top_1 += 1

  

  examine_loss += loss.item()


print("Top 1: " + str(top_1/len(database.reactions)))
print("Loss with penalty (only for larger than target): " + str(examine_loss/len(database.reactions)))
print("validity_rate: " + str(validity_rate/len(database.reactions)))
