import torch
import torch.nn as nn
import model

lines = []

inp_side = []
out_side = []

vocab = {}

with open("reaction_database.txt", "r") as file:
  for line in file:
    i, o = line.rstrip("\n").split(">>")[0], line.rstrip("\n").split(">>")[1]
    mi = []
    mo = []

    for k in i:
      if k not in vocab:
          vocab[k] = len(vocab)
      mi.append(vocab[k])   
    for k in o:
      if k not in vocab:
          vocab[k] = len(vocab)
      mo.append(vocab[k])   
 
    inp_side.append(torch.tensor(mi))
    out_side.append(torch.tensor(mo))
 

vol = len(vocab)

model = model.Seq2Seq(vol+2, 256, 4, 128, vol, vol+1)

criteron = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
 for u in range(len(inp_side)):
   src = inp_side[u]
   trg = torch.cat([torch.tensor([vol]),out_side[u], torch.tensor([vol+1])])
   optimiser.zero_grad()

   logits = model.train_step(src, trg)

   loss = criteron(logits, trg[1:])

   loss.backward()

   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

   optimiser.step()

 print(epoch) 
 print(epoch, loss.item())


with torch.no_grad():
  logits = model.forward(src)
  predicted_tokens = logits.argmax(dim=1)
  

print("Predicted tokens")
print(predicted_tokens)
