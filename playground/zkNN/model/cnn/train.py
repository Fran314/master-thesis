import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from common import *
from dataloader import prepare_dataloader


def train_model(model, train_loader, test_loader, device, qat=False):

    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    if qat:
        num_epochs = 20
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        num_epochs = 200
        optimizer = optim.SGD(model.parameters(), lr=5e-1, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=5e-6, T_max=200)
    
    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.5f} Train Acc: {:.5f} Eval Loss: {:.5f} Eval Acc: {:.5f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))
        scheduler.step()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet50', 'resnet101', 'vgg11'])

    args = parser.parse_args()

    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = f"{args.model}_cifar10.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(model=args.model, num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader()

    # Train model.
    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device)

    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)