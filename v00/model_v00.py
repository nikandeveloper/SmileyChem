import torch
import torch.nn as nn

class DeepRNN(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_n, embedding_size):
      super().__init__()

      self.embedding = nn.Embedding(input_size,embedding_size)
      
      layers = []
      
      layers.append(nn.Linear(embedding_size + hidden_size, hidden_size))
      layers.append(nn.ReLU())
      
      for _ in range(hidden_n - 2):
          layers.append(nn.Linear(hidden_size, hidden_size))
          layers.append(nn.ReLU())
      
      layers.append(nn.Linear(hidden_size, hidden_size))

      self.network = nn.Sequential(*layers)

  def forward(self, x, h):
      x = self.embedding(x)
      combined = torch.cat([x, h], dim=0)
      h = self.network(combined)

      return h


class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_n, embedding_size):
    super().__init__()
    self.rnn = DeepRNN(input_size, hidden_size, hidden_n, embedding_size)
  def forward(self, x, h):
    for t in range(x.size(0)):
      h = self.rnn(x[t], h)

    return h


class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_n, embedding_size):
    super().__init__()
    self.rnn = DeepRNN(input_size, hidden_size, hidden_n, embedding_size)
    self.output = nn.Linear(hidden_size, input_size)
  
  def forward(self, h, x):
    h = self.rnn(x, h)
    x = self.output(h)

    return h, x



class Seq2Seq(nn.Module):
  def __init__(
    self, 
    input_size,
    output_size,
    hidden_size,
    hidden_n,
    embedding_size,
    sos_token,
    eos_token
  ):
     
     super().__init__()
     self.encoder = Encoder(
        input_size,
        hidden_size,
        hidden_n,
        embedding_size
     )

     self.decoder = Decoder(
        input_size,
        hidden_size,
        hidden_n,
        embedding_size
     )

     self.hidden_size = hidden_size
     self.sos_token = sos_token
     self.eos_token = eos_token     


  def forward(self, src):
     h = torch.zeros(
        self.hidden_size,
        device=src.device 
     )

     h = self.encoder(src, h)
     decoder_input = torch.tensor(
        self.sos_token,
        dtype=torch.long,
        device=src.device)

     outputs = []

     for _ in range(100):
      h, logits = self.decoder(h, decoder_input)
      outputs.append(logits)

      decoder_input = logits.argmax(dim=0)

      if decoder_input.item() == self.eos_token:
        break

     return torch.stack(outputs, dim=0)

  def train_step(self, src, trg):
     h = torch.zeros(self.hidden_size, device=src.device)

     h = self.encoder(src, h)

     decoder_input = trg[0]

     outputs = []

     for t in range(1, trg.size(0)):
        
        h, logits = self.decoder(h, decoder_input)
        outputs.append(logits)

        decoder_input = trg[t]

     return torch.stack(outputs)
