import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def trainTeacherOnHparam(teacher_net, hparam, num_epochs,
                         train_loader, val_loader,
                         print_every=0,
                         fast_device=torch.device('cpu')):
    train_loss_list, train_acc_list, val_loss_list, val_acc_list= [], [], [], []
    teacher_net.dropout_input = hparam['dropout_input']
    teacher_net.dropout_hidden = hparam['dropout_hidden']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_dcay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])#调整学习率的方法
    for epoch in range(num_epochs):
        lr_scheduler.step()
        if epoch == 0:
            if val_loader is not None:
                val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                print('epoch: %d validation loss: %.3f validation accuracy: %.3f' %(epoch, val_loss, val_acc)#程序中没有val_loader
            for i, data in enumerate(train_loader, 0):
                X, y = data # X, y = (list) X, y 分别为list的两个元素
                X, y = X.to(fast_device), y.to(fast_device)
                loss, acc = trainStep(teacher_net, criterion, optimizer, X, y)
