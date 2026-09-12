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
PICKELED_REACTION_DATABASE_FILE_NAME = "reactions.db"
PTH_MODEL_NAME = "model.pth"




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))


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


model = md.Seq2Seq(vocab_size+2, 256, 4, 128, vocab_size, vocab_size+1)

if model_file.is_file():
  try:
    state_dict = torch.load("model.pth", weights_only=True)
    model.load_state_dict(state_dict)

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  print(f"File {PTH_MODEL_NAME} does not existing. Making new model...")




criteron = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()


for epoch in range(10):
  examine_loss = 0
  for u in range(len(database.reactions)):
   src = database.reactions[u].reactants_tokens
   trg = torch.cat([torch.tensor([vocab_size]),database.reactions[u].products_tokens, torch.tensor([vocab_size+1])])
   optimiser.zero_grad()

   src = src.to(device)
   trg = trg.to(device)

   logits = model.train_step(src, trg)

   loss = criteron(logits, trg[1:])

   examine_loss = loss.item()

   loss.backward()

   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

   optimiser.step()

  print(epoch, examine_loss)


torch.save(model.state_dict(), "model.pth")
