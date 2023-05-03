import torch
import numpy as np

def train(num_epochs, top_percent, train_loader, optimizer, criterion, net, buffer=True):
    # Training loop
    cumulative_gradients = {}
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        if buffer:
                            if name not in cumulative_gradients:
                                cumulative_gradients[name] = torch.zeros_like(param.grad.data)
                            cumulative_gradients[name] += param.grad.data
                            top_gradients = filter_top_gradients(cumulative_gradients[name], top_percent)
                            updated_mask = top_gradients != 0
                            param.grad.data = top_gradients
                            cumulative_gradients[name] = cumulative_gradients[name] * (~updated_mask)
                        else:
                            top_gradients = filter_top_gradients(param.grad.data, top_percent)
                            param.grad.data = top_gradients

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}")
    print('Finished Training')    

# Function to filter top x% gradients
def filter_top_gradients(grads, top_percent):
    flat_grads = torch.abs(grads.flatten())
    num_to_keep = int(np.ceil(top_percent * flat_grads.numel()))
    threshold = flat_grads.topk(num_to_keep).values[-1]
    mask = torch.abs(grads) >= threshold
    return grads * mask

# Testing the model
def test(test_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {num:d} test images: {acc:.2f} %'.format(num = len(test_loader.dataset), acc = (100 * correct / total)))


