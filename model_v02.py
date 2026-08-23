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
    outputs = []

    for t in range(x.size(0)):
      h = self.rnn(x[t], h)
      outputs.append(h)
    
    return outputs


class Attention(nn.Module):
   def __init__(self, hidden_size):
      super().__init__()

      self.hidden_size = hidden_size

      self.Lh = nn.Linear(hidden_size * 2, hidden_size)
      self.Lst = nn.Linear(hidden_size, hidden_size)
      self.Lv = nn.Linear(hidden_size, 1)

      self.Lcontext = nn.Linear(hidden_size * 2, hidden_size)


   def forward(self, hiddens, st):
      scores = []
      c_st = self.Lst(st)
      for i in range(len(hiddens)):
        c_h = self.Lh(hiddens[i])
        combined = c_st + c_h
        activated = torch.tanh(combined)
        score_i = self.Lv(activated)
        scores.append(score_i)      

      scores = torch.stack(scores)

      scores = scores.squeeze(1)  

      scores = torch.softmax(scores, dim=0)

      vector = torch.zeros_like(st)

      for k in range(len(hiddens)):
         vector += self.Lcontext(hiddens[k]) * scores[k]

      return vector   


class DecoderDeepRNN(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_n, embedding_size):
      super().__init__()

      self.embedding = nn.Embedding(input_size,embedding_size)
      
      layers = []
      
      layers.append(nn.Linear(embedding_size + hidden_size*2, hidden_size))
      layers.append(nn.ReLU())
      
      for _ in range(hidden_n - 2):
          layers.append(nn.Linear(hidden_size, hidden_size))
          layers.append(nn.ReLU())
      
      layers.append(nn.Linear(hidden_size, hidden_size))

      self.network = nn.Sequential(*layers)

  def forward(self, x, h, context):
      x = self.embedding(x)
      combined = torch.cat([x, h, context], dim=0)
      h = self.network(combined)

      return h


class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, hidden_n, embedding_size):
    super().__init__()
    self.rnn = DecoderDeepRNN(input_size, hidden_size, hidden_n, embedding_size)
    self.output = nn.Linear(hidden_size, input_size)

  
  def forward(self, h, x, context):
    h = self.rnn(x, h, context)
    x = self.output(h)

    return h, x



class Seq2Seq(nn.Module):
  def __init__(
    self, 
    input_size,
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

     self.attention = Attention(hidden_size)

     self.decoder = Decoder(
        input_size,
        hidden_size,
        hidden_n,
        embedding_size
     )

     self.hidden_size = hidden_size
     self.sos_token = sos_token
     self.eos_token = eos_token     

     self.decoding_init = nn.Linear(2 * hidden_size, hidden_size)

  def forward(self, src):
     h = torch.zeros(
        self.hidden_size,
        device=src.device 
     )


     all_hidden_normal = self.encoder(src, h)
     all_hidden_reverse = self.encoder(src.flip(0), h)
     all_hidden_reverse.reverse()

     all_hidden = []

     for i in range(len(all_hidden_normal)):
         combined = torch.cat((all_hidden_normal[i], all_hidden_reverse[i]), dim=0)
         all_hidden.append(combined)

     h = self.decoding_init(all_hidden[-1])

     decoder_input = torch.tensor(
        self.sos_token,
        dtype=torch.long,
        device=src.device)

     outputs = []

     for _ in range(100):
      context = self.attention(all_hidden, h)
      h, logits = self.decoder(h, decoder_input, context)
      outputs.append(logits)

      decoder_input = logits.argmax(dim=0)

      if decoder_input.item() == self.eos_token:
        break

     return torch.stack(outputs, dim=0)

  def train_step(self, src, trg):
     h = torch.zeros(self.hidden_size, device=src.device)

     
     all_hidden_normal = self.encoder(src, h)
     all_hidden_reverse = self.encoder(src.flip(0), h)
     all_hidden_reverse.reverse()

     all_hidden = []

     for i in range(len(all_hidden_normal)):
         combined = torch.cat((all_hidden_normal[i], all_hidden_reverse[i]), dim=0)
         all_hidden.append(combined)
         
     h = self.decoding_init(all_hidden[-1])

     decoder_input = trg[0]

     outputs = []

     for t in range(1, trg.size(0)):
        context = self.attention(all_hidden, h)
        h, logits = self.decoder(h, decoder_input, context)
        outputs.append(logits)

        decoder_input = trg[t]

     return torch.stack(outputs)
