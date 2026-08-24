import torch
import torch.nn as nn
import model
import tokeniser as t

t.ELEMENTS = [
    element
    for element in t.ELEMENTS
    if element not in t.ALIPHATIC
]

lines = []

inp_side = []
out_side = []

vocab_size = 0


with open("reaction_database.txt", "r") as file:
  for line in file:
    i, o = line.rstrip("\n").split(">>")[0], line.rstrip("\n").split(">>")[1]
    mi = []
    mo = []

    readeri = t.SmilesReader(i)
    readero = t.SmilesReader(o)

    mi = readeri.tokenise()
    mo = readero.tokenise()
    
    vocab_size = readeri.vocab_size()

    inp_side.append(torch.tensor(mi))
    out_side.append(torch.tensor(mo))
 


model = model.Seq2Seq(vocab_size+2, 256, 4, 128, vocab_size, vocab_size+1)

criteron = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
 for u in range(len(inp_side)):
   src = inp_side[u]
   trg = torch.cat([torch.tensor([vocab_size]),out_side[u], torch.tensor([vocab_size+1])])
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


torch.save(model.state_dict(), "model_v02.pth")
