import torch
import torch.nn as nn
import model_v03 as md
import tokeniser as t
import parser
import validation
import database as db
import pickle
import time
from pathlib import Path



CSV_RAW_REACTIONS_FILE_NAME = "raw_test.csv"
PICKELED_REACTION_DATABASE_FILE_NAME = "reactions.db"
PTH_MODEL_NAME = "model.pth"

CHECKPOINT_DIR = Path("checkpoint")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_LENGTH = 10 * 60 #assuming time in seconds


starting_epoch = 0
starting_reaction = 0
EPOCHS = 100





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


model = md.Seq2Seq(vocab_size+2, 256, 3, 256, vocab_size, vocab_size+1)

if model_file.is_file():
  try:
    state_dict = torch.load("model.pth", weights_only=True)
    model.load_state_dict(state_dict)

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  print(f"File {PTH_MODEL_NAME} does not existing. Making new model...")

model.to(device)


criteron = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()

last_checkpoint = time.time()


for epoch in range(starting_epoch, EPOCHS):
  examine_loss = 0
  for u in range(starting_reaction, len(database.reactions)):
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

   if time.time() - last_checkpoint >= CHECKPOINT_LENGTH:
     check_point_file = CHECKPOINT_DIR / f"model_epoch_{epoch}_reaction_{u}.pth"
     saved_data = {"epoch": epoch, "reaction": u, "model_state_dict": model.state_dict(),  "optimiser_state_dict": optimiser.state_dict(), "loss": loss.item()}

     torch.save(saved_data, check_point_file)

     print(f"saved checkpoint: {check_point_file}")
     print(f"Loss: {loss.item()}")

     last_checkpoint = time.time()




torch.save(model.state_dict(), "model.pth")
