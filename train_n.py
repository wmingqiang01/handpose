import os
import argparse
import torch
import torch.nn as nn
from hand_data_iter.datasets import *
from models.mobilenetv2 import MobileNetV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from loss.loss import got_total_wing_loss


def trainer(ops):

    model_ = MobileNetV2(num_classes=ops.num_classes, dropout_factor=ops.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_ = model_.to(device)

    dataset = LoadImagesAndLabels(ops=ops)
    dataloader = DataLoader(dataset, batch_size=ops.batch_size, num_workers=3, shuffle=True)

    optimizer = torch.optim.Adam(model_.parameters(), lr=0.001, betas=(0.9, 0.99))
    criterion = nn.MSELoss()

    best_loss = float('inf')
    loss_values = []

    for epoch in range(ops.epochs):
        model_.train()
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{ops.epochs}")
        for imgs_, pts_ in pbar:
            if torch.cuda.is_available():
                imgs_, pts_ = imgs_.cuda(), pts_.cuda()

            output = model_(imgs_.float())
            loss = criterion(output, pts_.float())
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({'loss': sum(losses) / len(losses)})

        avg_loss = sum(losses) / len(losses)
        loss_values.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model_.state_dict(), os.path.join(ops.model_exp, f'{ops.model}-best_model.pth'))

    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.savefig(os.path.join(ops.model_exp, 'loss_plot.png'))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Hand Train')
    parser.add_argument('--model_exp', type=str, default='./model_exp')
    parser.add_argument('--dropout', type=float, default = 0.5) # dropout
    parser.add_argument('--epochs', type=int, default = 300) # 训练周期
    parser.add_argument('--num_classes', type=int , default = 42) #  landmarks 个数*2
    parser.add_argument('--train_path', type=str, default="./handpose_datasets_v1/", help='datasets')
    parser.add_argument('--batch_size', type=int, default=16)
    # ... other arguments ...
    args = parser.parse_args()
    trainer(ops=args)
    print('Training finished.')
