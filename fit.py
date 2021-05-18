import argparse
import os.path

import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.model import VAE
from src.train_helper import VAE_loss


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Downloads the ISIC 2016 Challenge training dataset')
    parser.add_argument('--dataset-path', '-d', type=str, help="path to the folder with images")
    parser.add_argument('--output', '-o', type=str, help="path to the output folder where the report is saved")

    parser.add_argument('--features', type=int, default=16, help="dimensionality of the representation")
    parser.add_argument('--lr', type=int, default=0.0001, help="the learning rate used for training")
    parser.add_argument('--epoch', type=int, default=2, help="number of epochs to train the VAE for")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size used for training")
    parser.add_argument('--train-val-split', type=float, default=0.9,
                        help="split ratio of the data to training and validation set")

    args = parser.parse_args()
    return args


def get_loaders(dataset_path, batch_size=32, train_val_split=.9):
    """ Returns two DataLoader object for the training data and validation data respectively.

    :param dataset_path: path to the datasets to be loaded.
    :param batch_size: batch size used for training
    :param train_val_split: if .9 then 90% of the data is used for training and 10% used for testing.
    :return: train_loader and val_loader
    """

    transform = transforms.Compose([
        transforms.Resize(size=(28,28)),
        transforms.ToTensor(),
    ])

    data = torchvision.datasets.ImageFolder(dataset_path,transform=transform)  # load all the data.
    n = len(data)  # total number of examples
    n_train = int(train_val_split * n)
    train_set = torch.utils.data.Subset(data, range(n_train))  # take the train_val_split percet of the data
    val_set = torch.utils.data.Subset(data, range(n_train, n))  # take the rest for validation

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    msg = "Loaded {} training and {} validation images!"
    print(msg.format(train_loader.sampler.num_samples, len(val_loader.sampler)))

    return train_loader, val_loader


def fit(model, data_loader, optimizer, device):
    bce_losses = 0
    total_losses = 0
    model.train()
    criterion = nn.BCELoss(reduction='sum')
    for i, data in tqdm(enumerate(data_loader)):
        data, _ = data
        data = data.to(device)
        #data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = VAE_loss(bce_loss, mu, logvar)
        bce_losses += bce_loss.item()
        total_losses += loss.item()
        loss.backward()
        optimizer.step()

    return bce_losses / len(data_loader.dataset), total_losses / len(data_loader.dataset)


def val(model, data_loader, device):
    bce_losses = 0
    total_losses = 0
    model.eval()
    criterion = nn.BCELoss(reduction='sum')
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            data, _ = data
            data = data.to(device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = VAE_loss(bce_loss, mu, logvar)
            bce_losses += bce_loss.item()
            total_losses += loss.item()

    return bce_losses / len(data_loader.dataset), total_losses / len(data_loader.dataset)


def export_reconstruction_as_image(model, data_loader, device, output):
    # save the last batch input and output of every epoch
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            data, _ = data
            data = data.to(device)
            reconstruction, mu, logvar = model(data)


            both = torch.cat((data.view(*data.size()),
                              reconstruction.view(*data.size())))
            save_image(both.cpu(), output, nrow=data.size()[0])


def plot_performace(bce_loss_train, total_loss_train, bce_loss_val, total_loss_val, output):
    datas = [bce_loss_train, total_loss_train, bce_loss_val, total_loss_val]
    labels = ["bce_loss_train", "total_loss_train", "bce_loss_val", "total_loss_val"]

    for data, label in zip(datas, labels):
        x = range(len(data))
        fig, ax = plt.subplots()
        ax.plot(x, data)

        ax.set(xlabel='epoch', ylabel=label,
               title=label)
        ax.grid()

        fig.savefig(os.path.join(output, label+".svg"))
    pass

def main(dataset_path, output, features, lr, epoch, batch_size=32, train_val_split=.9):
    train_loader, val_loader = get_loaders(dataset_path, batch_size=32, train_val_split=.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(features=features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    bce_loss_train = []
    total_loss_train = []
    bce_loss_val = []
    total_loss_val = []
    for e in range(epoch):
        msg = "Epoch: {}/{}".format(e, epoch)
        print(msg)
        _bce_loss, _total_loss = fit(model, train_loader, optimizer, device)
        bce_loss_train.append(_bce_loss)
        total_loss_train.append(_total_loss)

        _bce_loss, _total_loss = val(model, val_loader, device)
        bce_loss_val.append(_bce_loss)
        total_loss_val.append(_total_loss)

        output_img = os.path.join(output, "{}.png".format(e))
        export_reconstruction_as_image(model, val_loader, device, output_img)

        report_msg = "bce train loss: {} \t total train loss: {}".format(bce_loss_train[-1], total_loss_train[-1])
        print(report_msg)
        report_msg = "bce val loss: {} \t total val loss: {}".format(bce_loss_val[-1], total_loss_val[-1])
        print(report_msg)

    plot_performace(bce_loss_train, total_loss_train, bce_loss_val, total_loss_val, output)

if __name__ == "__main__":
    args = parseargs()
    main(**args.__dict__)
