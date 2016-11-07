#Toy example of seq2seq modeling using vanilla RNN 
# h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bxh + bhh)
# y(t) = Wyh*h(t) + byh 
# x(t) is n dimensional embedding for one hot vector for vocabulary of length m 

import numpy as np
import os
import sys

