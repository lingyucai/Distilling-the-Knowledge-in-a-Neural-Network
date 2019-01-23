import numpy as np
import itertools
import os
import torch
import torch.nn.functional as F
import networks
import utils

import torchvision
import torchvision.transforms as transforms

cpu_device = torch.deviece('cuda')
fast_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = True

def reproducibilitySeed():
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed) # torch.manual_seed(args.seed) 为CPU设置种子用于生成随机数，以使得结果是确定的
    #在定义网络的时候，如果层内有Variable,那么用nn定义，反之，则用nn.functional定义
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reproducibilitySeed()

mnist_image_shape = (28, 28)
random_pad_size = 2
# 通过随机以最大两个像素在四个方向旋转图像实现图像增强
transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(mnist_image_shape, random_pad_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5), (0.5, 0.5))
                ]
            )
transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5), (0.5,0.5))
                ]
            )

train_val_dataset = torchvision.datasets.MNIST(root='./MINIST_dataset/', train=True,
                                               download=True, transforms=transform_train)
test_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=False,
                                            download=True, transform=transform_test)
num_train = int(1.0 * len(train_val_dataset) * 95 / 100)
num_val = len(train_val_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

batch_size = 128
train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=128, shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

checkpoints_path = 'checkpoints_teacher/'
if not os.path.exits(checkpoints_path):
    os.mkdir(checkpoints_path)# 创建制定名字的目录

num_epochs = 60
print_every = 100
learning_rates = [1e-2]
learning_rate_decays = [0.95]
weight_decays = [1e-5]
momentums = [0.9]
dropout_probabilities = [(0.0, 0.0)]
hparams_list = []
for hparam_tuple in itertools.product(dropout_probabilities, weight_decays, learning_rate_decays,
                                      momentums, learning_rates):
    hparam = {}
    hparam['dropout_input'] = hparam_tuple[0][0]
    hparam['dropout_hidden'] = hparam_tuple[0][1]
    hparam['weight_decay'] = hparam_tuple[1]
    hparam['lr_decay'] = hparam_tuple[2]
    hparam['momentum'] = hparam_tuple[3]
    hparam['lr'] = hparam_tuple[4]
    hparams_list.append(hparam)

result = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.hparamToString(hparam))#字典转为string输出
    reproducibilitySeed()
    teacher_net = networks.TeacherNetwork()
    teacher_net = teacher_net.to(fast_device)
    hparam_tuple = utils.hparamDictToTuple(hparam)#将dict的value转为tuple，丢掉key tuple([v for k, v in sorted(hparam.items())]) 先转list再转tuple
    results[hparam_tuple] = utils.trainTeacherOnHparam(teacher_net, hparam, num_epochs,
                                                       train_val_loader, None,
                                                       print_every=print_every,
                                                       fast_device=fast_device)
