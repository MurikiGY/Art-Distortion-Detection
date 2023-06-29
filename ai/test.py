import cnn
import torch
from torchsummary import summary

device = "cpu"
new_model = cnn.Neural_Network(1, 2).to(device)

print("New NN")
summary(new_model, (1, 256, 256))
