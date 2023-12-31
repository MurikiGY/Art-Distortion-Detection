import torch
import dataset
import cnn
import torchvision
import matplotlib.pyplot as plt
import time

from torch.optim import Adam
from torch.utils.data import DataLoader


start_time = time.time()

#selecionando dispositivo de hardware

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

img_size = 256
resizer = torchvision.transforms.Resize((img_size, img_size))
grayscaler = torchvision.transforms.Grayscale()

def treat_img(img):
    img = resizer(img)
    img = grayscaler(img)
    img = torchvision.transforms.functional.convert_image_dtype(img, torch.float32)
    return img

#ajeitando dados

data = dataset.ImageDataset(
        "data/",
        transform = treat_img,
        target_transform = lambda y: torch.zeros(
            2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
    )

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

BATCH_SIZE = 64
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


#print(data[0])


#Parte do aprendizado de máquina

trainSteps = len(train_dataloader.dataset) // BATCH_SIZE
valSteps = len(test_dataloader.dataset) // BATCH_SIZE

model = cnn.Neural_Network(1, 2).to(device)

history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
INIT_LR = 0.001
EPOCHS = 10

opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = torch.nn.CrossEntropyLoss()

print("Training...")

for e in range(EPOCHS):
    print("Epoch number ", e, " of training")
    aux_time = time.time()

    model.train()

    tTrainLoss = 0
    tValLoss = 0

    trainCorrect = 0
    valCorrect = 0

    for (x, y) in train_dataloader:
        (x, y) = (x.to(device), y.to(device))

        pred = model(x)
        loss = lossFn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tTrainLoss += loss

        for i in range(len(y)):
            if torch.argmax(pred[i]) == torch.argmax(y[i]):
                trainCorrect += 1

    with torch.no_grad():
        model.eval()

        for (x, y) in test_dataloader:
            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            tValLoss += lossFn(pred, y)


            for i in range(len(y)):
                if torch.argmax(pred[i]) == torch.argmax(y[i]):
                    valCorrect += 1


    history["train_acc"].append(trainCorrect / len(train_dataloader.dataset))
    history["test_acc"].append(valCorrect / len(test_dataloader.dataset))

    history["train_loss"].append(tTrainLoss / trainSteps)
    history["test_loss"].append(tValLoss / valSteps)

    print("Finished epoch ", e)
    print(" Train loss: {:.8f}, Train accuracy: {:.8f}".format(history["train_loss"][-1], history["train_acc"][-1]))
    print(" Test loss: {:.8f}, Test accuracy: {:.8f}".format(history["test_loss"][-1], history["test_acc"][-1]))

    print(" Took {:.5f} seconds".format(time.time()-aux_time))

    print("Saving...")
    torch.save(model, "model.pth")

    #plot

    with torch.no_grad():
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["test_loss"], label="val_loss")
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["test_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("graph.png")

print(history)
print("Finished training in {:.5f} seconds".format(time.time()-aux_time))





