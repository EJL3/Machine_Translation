# Commented out IPython magic to ensure Python compatibility.
##Imports
# %pip install rouge_score
# %pip install datasets
import io
from datasets import load_dataset, load_dataset_builder, inspect_dataset
import re
import unicodedata
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import math
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import numpy as np
from torchtext.data.metrics import bleu_score
from rouge_score import rouge_scorer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##Datasets
datasetRu = load_dataset("un_pc", "en-ru", split='train[:20%]')

##to remember how datasets work
#for i in range (10):
  #print(datasetRu['translation'][i]['en'])

##Lang class
SOS_token = 0
EOS_token = 1

class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
    self.n_words = 3  # Count SOS and EOS and PAD

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1


def unicodeToAscii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
  )


def normalizeString(s,lang=0):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  if lang==('en'):
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  if lang==('ru'):
    s = re.sub(r"[^а-яА-Я.!?]+", r" ", s)
  return s


def readLangs(lang1, lang2, dataset, reverse=False):

  input_lang = Lang(lang1)
  output_lang = Lang(lang2)
  pairs=[]
  data=dataset['translation']
  if reverse:
    for pair in data:
      newPair=[normalizeString(pair[lang2],lang2),normalizeString(pair[lang1],lang1)]
      pairs.append(newPair)
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  else:
    for pair in data:
      newPair=[normalizeString(pair[lang1],lang1),normalizeString(pair[lang2],lang2)]
      pairs.append(newPair)
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
  return input_lang, output_lang, pairs

##Filtering sentenses
MAX_LENGTH = 10
MIN_LENGTH = 3

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ", "it is "
)


def filterPair(p, lang1, lang2, reverse):
  return len(p[0].split(' ')) < MAX_LENGTH and \
         len(p[1].split(' ')) < MAX_LENGTH and \
         len(p[0].split(' ')) > MIN_LENGTH and \
         len(p[1].split(' ')) > MIN_LENGTH

def filterPairs(pairs, lang1, lang2, reverse):
  return [pair for pair in pairs if filterPair(pair, lang1, lang2, reverse)]

##Final data preporation
def prepareData(lang1, lang2, dataset, reverse=False):
  input_lang, output_lang, pairs = readLangs(lang1, lang2, dataset, reverse)
  print("Read %s sentence pairs" % len(pairs))
  pairs = filterPairs(pairs, lang1, lang2, reverse)
  print("Trimmed to %s sentence pairs" % len(pairs))
  print("Counting words...")
  pairs=random.sample(pairs, 120000)
  for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])
  print("Counted words:")
  print(lang1, input_lang.n_words)
  print(lang2, output_lang.n_words)
  return input_lang, output_lang, pairs


input_lang, output_lang, pairs =prepareData('en', 'ru', datasetRu)
trPairs=(pairs[:10000])
tsPairs=(pairs[100000:])

def plot_lang(lang, top_k=100):
  words = list(lang.word2count.keys())
  words.sort(key=lambda w: lang.word2count[w], reverse=True)
  print(words[:top_k])
  count_occurences = sum(lang.word2count.values())

  accumulated = 0
  counter = 0

  while accumulated < count_occurences * 0.8:
    accumulated += lang.word2count[words[counter]]
    counter += 1

  print(f"The {counter * 100 / len(words)}% most common words "
        f"account for the {accumulated * 100 / count_occurences}% of the occurrences")
  plt.bar(range(100), [lang.word2count[w] for w in words[:top_k]])
  plt.show()

plot_lang(input_lang)
plot_lang(output_lang)

##Encoder-Decoder model
class EncoderRNN(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

  def forward(self, input, hidden):
    embedded = self.embedding(input)#.view(1, 1, -1)
    output = embedded
    output, hidden = self.gru(output, hidden)
    return output, hidden

  def initHidden(self, batch_size):
    return torch.zeros(1, batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
  def __init__(self, hidden_size, output_size):
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(output_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    self.out = nn.Linear(hidden_size, output_size)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, input, hidden):
    output = self.embedding(input)
    output = F.relu(output)
    output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output))
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)

def to_train(input_lang, output_lang, pairs, max_len=MAX_LENGTH+2):
  x_input = []
  x_output = []
  target = []
  for i, o in pairs:
    s_i = [2] * max_len + [0] + [input_lang.word2index[w] for w in i.split(" ")] + [1]
    s_o = [0] + [output_lang.word2index[w] for w in o.split(" ")] + [1] + [2] * max_len
    s_to = s_o[1:] + [2]
    x_input.append(s_i[-max_len:])
    x_output.append(s_o[:max_len])
    target.append(s_to[:max_len])
  return x_input, x_output, target

x_input, x_partial, y = to_train(input_lang, output_lang, trPairs)


print('Representation of an input sentece:')
print(x_input[0])
print(' '.join([input_lang.index2word[w] for w in x_input[0]]))
print('\nRepresentation of an partial sentece:')
print(x_partial[0])
print(' '.join([output_lang.index2word[w] for w in x_partial[0]]))
print('\nRepresentation of an target sentece:')
print(y[0])
print(' '.join([output_lang.index2word[w] for w in y[0]]))

def predict(encoder, decoder, input, output):
  _, hidden = encoder(input, encoder.initHidden(input.shape[0]))
  out, _ = decoder(output, hidden)
  return out

def train(encoder, decoder, loss, input, output, target, learning_rate=0.001, epochs=10, batch_size=32):

  plot_losses = []
  plot_full_losses = []

  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

  for _ in tqdm(range(epochs)):
    c_input, c_output, c_target = shuffle(input, output, target)
    c_input = torch.tensor(c_input, dtype=torch.long, device=device)
    c_output = torch.tensor(c_output, dtype=torch.long, device=device)
    c_target = torch.tensor(c_target, dtype=torch.long, device=device)
    acc_loss = 0
    for i in range(0, c_target.shape[0], batch_size):
      c_batch_size = c_target[i:i+batch_size, 1].shape[0]
      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()

      out = predict(encoder, decoder, c_input[i:i+batch_size, ...], c_output[i:i+batch_size, ...])
      #Reshapes the output and target to use the expected loss format.
      # N x Classes for the output
      # N for the targets
      # Where N is the batch size
      out = out.reshape(c_batch_size * c_input.shape[1], -1)
      r_target = c_target[i:i+batch_size, ...].reshape(c_batch_size * c_input.shape[1])

      c_loss = loss(out, r_target)
      # Mask the errors for padding as they are not usefull!
      valid = torch.where(r_target == 2, 0, 1)
      c_loss = c_loss * valid
      c_loss = torch.sum(c_loss) #/ torch.sum(valid)

      c_loss.backward()

      encoder_optimizer.step()
      decoder_optimizer.step()
      plot_full_losses.append(c_loss.detach().numpy())
      acc_loss += c_loss.detach().numpy()
    plot_losses.append(acc_loss /math.ceil(c_target.shape[0] / batch_size))
  return plot_losses, plot_full_losses

hidden_size = 300
num_epochs = 50
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words)
epoch_error, batch_error = train(encoder, decoder,
                                 nn.NLLLoss(reduction='none'),
                                 x_input, x_partial, y,
                                 epochs=num_epochs)

##Loss graphs
plt.plot(batch_error)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('minibatch')
plt.show()

plt.plot(epoch_error)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

p = predict(encoder, decoder, torch.tensor([x_input[40]],
                                           dtype=torch.long,
                                           device=device),
            torch.tensor([x_partial[40]], dtype=torch.long, device=device))

p = p.detach().numpy()

print(np.argmax(p, axis=-1))
print(x_partial[40])

def gen_translation(encoder, decoder, text, input_lang, output_lang,
                    max_len=MAX_LENGTH+2):

  text =  [2] * max_len + [0] + [input_lang.word2index[w] for w in text.split(" ")] + [1]
  text = torch.tensor([text[-max_len:]], dtype=torch.long, device=device)
  out = [0] + [2] * max_len
  out = [out[:max_len]]
  for i in range(1, max_len):
    pt_out =torch.tensor(out, dtype=torch.long, device=device)
    p = predict(encoder, decoder, text, pt_out).detach().numpy()
    out[0][i] = np.argmax(p, axis=-1)[0, i-1]
    if np.argmax(p, axis=-1)[0, i-1] == 1:
      break

  return ' '.join([output_lang.index2word[idx] for idx in out[0]])

  gen_translation(encoder1, decoder1, pairs1[40][0], input_lang1, output_lang1)

  gen_translation(encoder2, decoder2, pairs2[40][0], input_lang2, output_lang2)

for i in range(40):
  print('> {}'.format(tsPairs[i][0]))
  print('= {}'.format(tsPairs[i][1]))
  print('< {}'.format(gen_translation(encoder, decoder,
                                      tsPairs[i][0],
                                      input_lang,
                                      output_lang)))
  print('*' * 40)

prPairs=[]
for pair in tsPairs:
  prPairs.append(gen_translation(encoder, decoder,
                                      tsPairs[pair][0],
                                      input_lang,
                                      output_lang))

##Scores
resValue=0
for i in range(200):
  pred=[prPairs[i].split(' ')]
  target=[[tsPairs[i][1].split(' ')]]
  # pred=[datasetRu['translation'][i]['ru'].split(' ')]
  # target=[[datasetRu['translation'][i]['ru'].split(' ')]]
  value= bleu_score(pred, target)
  resValue+=value
resValue=resValue/200
print('BLEUscore = ', resValue) # prints 0

score=0
recal=0
fmeasure=0
scorer=rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
for i in range(200):
  pred=prPairs[1]
  target=tsPairs[i][1]
  #pred=datasetRu['translation'][i]['ru']
  #target=datasetRu['translation'][i]['ru']
  scores=scorer.score(pred, target)
  s=scores['rougeL'][0]
  score+=s
  r=scores['rougeL'][1]
  recal+=r
  f=scores['rougeL'][2]
  fmeasure+=f
score=score/200
recal=recal/200
fmeasure=fmeasure/200
print('Score = ', score, 'Recal = ', recal, 'Fmeasure = ', fmeasure) # prints 0 for all
