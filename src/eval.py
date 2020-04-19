import torch

def evaluation(net1, net2, test_loader1, test_loader2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('working on evaluation ...')

    # eval residents
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader1):
            inputs, labels = data
            labels = labels[:, 0]
            if torch.cuda.is_available():
                outputs = net1(inputs.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
            else:
                outputs = net1(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    acc = correct / total
    print('resident precision: {:.4f}'.format(acc))

    # eval activity
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader2):
            inputs, labels = data
            labels = labels[:, 1]
            if torch.cuda.is_available():
                outputs = net2(inputs.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
            else:
                outputs = net2(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    acc = correct / total
    print('activity precision: {:.4f}'.format(acc))

