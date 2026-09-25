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
TRAINED_MODEL_FILE_NAME = "model.pth"

BUCKET_SIZE = 100
BATCH_SIZE = 64


CHECKPOINT_DIR = Path("checkpoint")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_LENGTH = 4 * 60 #assuming time in seconds


starting_epoch = 0
starting_batch_key = 0
starting_batch_element = 0
EPOCHS = 100

ignoring_index = -100



"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
"""

device = torch_xla.device()

folder = Path(__file__).parent

checkpoint = folder / "checkpoint"

files = [file.name for file in checkpoint.iterdir() if file.is_file()]

best_name = ""

files_nums = []

best_nums = [-1, -1, -1]


for name in files:
  #three numbers epoch, batch_key, batch_element

  parts = name.removesuffix(".pth").split("_")

  file_epoch = int(parts[2])
  file_bacth_key = int(parts[5])
  file_batch_element = int(parts[8])

  nums = (file_epoch, file_bacth_key, file_batch_element)

  if nums > best_nums:
    best_nums = nums
    best_name = name



PTH_MODEL_NAME = best_name




reactions_file = folder / PICKELED_REACTION_DATABASE_FILE_NAME
model_file = folder / "checkpoint" / PTH_MODEL_NAME

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

  with open(reactions_file, "wb") as file:
   
    pickle.dump(database ,file)





model = md.Seq2Seq(vocab_size+2, 256, 3, 256, vocab_size, vocab_size+1)
optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

if model_file.is_file():
  try:
    state_dict = torch.load(model_file, weights_only=False)
    model.load_state_dict(state_dict["model_state_dict"])
    model = model.to(device)
    optimiser.load_state_dict(state_dict["optimiser_state_dict"])
    starting_batch_key = state_dict["batch_key"]
    starting_batch_element = state_dict["batch_number"]
    starting_epoch = state_dict["epoch"]

  except:
    raise ValueError(f"The state dict in file {PTH_MODEL_NAME} is corrupted")

else:
  model = model.to(device)
  print(f"File {PTH_MODEL_NAME} does not existing. Making new model...")





criteron = nn.CrossEntropyLoss(ignore_index=ignoring_index)


model.train()


last_checkpoint = time.time()


for epoch in range(starting_epoch, EPOCHS):

 examine_loss = 0
  
 for key in database.bucket_dict_reactant.keys():

  if epoch == starting_epoch and starting_batch_key > key:
   continue


  for i, batch_reac in enumerate(database.bucket_dict_reactant[key]):  

    if epoch == starting_epoch and (starting_batch_key == key  and starting_batch_element >= i):
      continue

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
     check_point_file = CHECKPOINT_DIR / f"model_epoch_{epoch}_batch_key_{key}_batch_number_{i}.pth"

     cpu_state_dict = {name: p.detach().cpu() for name, p in model.state_dict().items()}

     saved_data = {"epoch": epoch, "batch_key": key, "batch_number": i, "model_state_dict": cpu_state_dict,  "optimiser_state_dict": optimiser.state_dict(), "loss": loss.item()}

     torch.save(saved_data, check_point_file)

     print(f"saved checkpoint: {check_point_file}")
     print(f"Loss: {loss.item()}")

     last_checkpoint = time.time()


 print(epoch, examine_loss/len(database.bucket_dict_reactant))



cpu_state_dict = {name: p.detach().cpu() for name, p in model.state_dict().items()}

saved_data = {"epoch": EPOCHS, "model_state_dict": cpu_state_dict,  "optimiser_state_dict": optimiser.state_dict(), "loss": examine_loss}
torch.save(saved_data, TRAINED_MODEL_FILE_NAME)
