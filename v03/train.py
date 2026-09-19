import torch
import torch.nn as nn
import model_v03 as md
import tokeniser as t
import parser
import validation
import database as db
import pickle
import time
import torch_xla
import torch_xla.core.xla_model as xlam
from pathlib import Path



CSV_RAW_REACTIONS_FILE_NAME = "raw_train.csv"
PICKELED_REACTION_DATABASE_FILE_NAME = "reactions.db"
PTH_MODEL_NAME = "model.pth"

BUCKET_SIZE = 10
BATCH_SIZE = 200


CHECKPOINT_DIR = Path("checkpoint")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_LENGTH = 10 * 60 #assuming time in seconds


starting_epoch = 0
starting_batch = 0
EPOCHS = 100

ignoring_index = -100



"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
"""

device = torch_xla.device()


reactions_file = Path(__file__).parent / PICKELED_REACTION_DATABASE_FILE_NAME
model_file = Path(__file__).parent / PTH_MODEL_NAME

vocab_size = t.SmilesReader.vocab_size()



if reactions_file.is_file():

  with open(reactions_file, "rb") as file:

    database = pickle.load(file)

else:

  database = db.Database(CSV_RAW_REACTIONS_FILE_NAME, BUCKET_SIZE, BATCH_SIZE, db.ignore)
  database.load_data()
  database.canonicalise_mapped()
  database.tokenise()
  database.add_marking_codes(vocab_size + 1, vocab_size)
  database.bucket_batch(-100)

  with open(PICKELED_REACTION_DATABASE_FILE_NAME, "wb") as file:
   
    pickle.dump(database ,file)





model = md.Seq2Seq(vocab_size+2, 256, 3, 256, vocab_size, vocab_size+1)

if model_file.is_file():
  try:
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict["model_state_dict"])

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  print(f"File {PTH_MODEL_NAME} does not existing. Making new model...")


model = model.to(device)


criteron = nn.CrossEntropyLoss(ignore_index=ignoring_index)

optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()


last_checkpoint = time.time()


for epoch in range(EPOCHS):
  
 examine_loss = 0
  
 for key in database.bucket_dict_reactant.keys():
  for i, batch_reac in enumerate(database.bucket_dict_reactant[key]):  
   src = batch_reac
   trg = database.bucket_dict_product[key][i]
   
   optimiser.zero_grad()

   src = src.to(device)
   trg = trg.to(device)

   logits = model.train_step(src, trg)

   loss = criteron(logits.transpose(1,2), trg[:, 1:])

   examine_loss += loss.item()

   loss.backward()

   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

   xlam.optimizer_step(optimiser)
   torch_xla.sync()

   if time.time() - last_checkpoint >= CHECKPOINT_LENGTH:
     check_point_file = CHECKPOINT_DIR / f"model_epoch_{epoch}_batch_key_{key}.pth"
     saved_data = {"epoch": epoch, "batch_key": key, "model_state_dict": model.state_dict(),  "optimiser_state_dict": optimiser.state_dict(), "loss": loss.item()}

     torch.save(saved_data, check_point_file)

     print(f"saved checkpoint: {check_point_file}")
     print(f"Loss: {loss.item()}")

     last_checkpoint = time.time()


 print(epoch, examine_loss/len(database.bucket_dict_reactant))




saved_data = {"epoch": EPOCHS, "model_state_dict": model.state_dict(),  "optimiser_state_dict": optimiser.state_dict(), "loss": examine_loss}
torch.save(saved_data, model_file)
