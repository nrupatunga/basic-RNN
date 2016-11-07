#Toy example of seq2seq modeling using vanilla RNN 
# h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bxh + bhh)
# y(t) = Wyh*h(t) + byh 
# x(t) is n dimensional embedding for one hot vector for vocabulary of length m 

import numpy as np
import os
import sys

class VanillaRNN:
  define __init__(self,txtFile,num_hidden_units):
    f = open(txtFile,'r')
    data = f.read()
    chars = list(set(data))
    num_data = len(data)
    num_chars = len(chars) 
    char_to_index = {i:c for i,c in enumerate(chars)}
    index_to_char = {c:i for i,c in enumerate(chars)}
    self.num_hidden_units = num_hidden_units
    self.vocab_size = num_chars
    self.whh = np.random.randn(num_hidden_units,num_hidden_units)*0.01
    self.wxh = np.random.randn(self.vocab_size,num_hidden_units)*0.01
    self.wyh = np.random.randn(self.vocab_size,num_hidden_units)*0.01
    self.bhh = np.random.randn(num_hidden_units,1)
    self.byh = np.random.randn(self.vocab_size,1)
    
    
