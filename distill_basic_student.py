import numpy as np
import math

import matplotlib.pyplot as plt

import itertools
import os

import torch
import torch.nn.functional as F

import networks
import utils

import torchvision
import torchvision.transforms as transforms



cpu_device = torch.device('cpu')
fast_device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = True
'''
gpu_id = 2

if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    fast_device = torch.device('cuda:' + str(gpu_id))
'''

def reproducibilitySeed():
    """
    Ensure reproducibility of results; Seeds to 0
    """
    torch_init_seed = 0
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = 0
    np.random.seed(numpy_init_seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

reproducibilitySeed()
transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5), (0.5, 0.5))
                ]
            )

train_val_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=True,
                                            download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=False,
                                            download=True, transform=transform)

num_train = int(1.0 * len(train_val_dataset) * 95 / 100)
num_val = len(train_val_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [num_train, num_val])

batch_size = 128
train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=128, shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

checkpoints_path_teacher = 'checkpoints_teacher/'
checkpoints_path_student = 'checkpoints_student/'
if not os.path.exists(checkpoints_path_student):
    os.makedirs(checkpoints_path_student)

# set the hparams used for training teacher to load the teacher network
learning_rates = [1e-2]
learning_rate_decays = [0.95]
weight_decays = [1e-5]
momentums = [0.9]
# keeping dropout input = dropout hidden
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

load_path = checkpoints_path_teacher + utils.hparamToString(hparams_list[0]) + '_final.tar'
teacher_net = networks.TeacherNetwork()
teacher_net.load_state_dict(torch.load(load_path, map_location=fast_device)['model_state_dict'])
teacher_net = teacher_net.to(fast_device)

# Calculate teacher test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(teacher_net, test_loader, fast_device)
print('teacher test accuracy: ', test_accuracy)

num_epochs = 60
print_every = 100

temperatures = [1]    # temperature for distillation loss
# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
# loss = alpha * st + (1 - alpha) * tt
alphas = [0.0]
learning_rates = [1e-2]
learning_rate_decays = [0.95]
weight_decays = [1e-5]
momentums = [0.9]
# No dropout used
dropout_probabilities = [(0.0, 0.0)]
hparams_list = []
for hparam_tuple in itertools.product(alphas, temperatures, dropout_probabilities, weight_decays, learning_rate_decays,
                                        momentums, learning_rates):
    hparam = {}
    hparam['alpha'] = hparam_tuple[0]
    hparam['T'] = hparam_tuple[1]
    hparam['dropout_input'] = hparam_tuple[2][0]
    hparam['dropout_hidden'] = hparam_tuple[2][1]
    hparam['weight_decay'] = hparam_tuple[3]
    hparam['lr_decay'] = hparam_tuple[4]
    hparam['momentum'] = hparam_tuple[5]
    hparam['lr'] = hparam_tuple[6]
    hparams_list.append(hparam)

results_no_distill = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.hparamToString(hparam))
    reproducibilitySeed()
    student_net = networks.StudentNetwork()
    student_net = student_net.to(fast_device)
    hparam_tuple = utils.hparamDictToTuple(hparam)
    results_no_distill[hparam_tuple] = utils.trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs,
                                                                    train_val_loader, None,
                                                                    print_every=print_every,
                                                                    fast_device=fast_device)
    save_path = checkpoints_path_student + utils.hparamToString(hparam) + '_final.tar'
    torch.save({'results' : results_no_distill[hparam_tuple],
                'model_state_dict' : student_net.state_dict(),
                'epoch' : num_epochs}, save_path)

# Calculate student test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(student_net, test_loader, fast_device)
print('student test accuracy (w/o distillation): ', test_accuracy)

# plt.rcParams['figure.figsize'] = [10, 5]
weight_decay_scatter = ([math.log10(h['weight_decay']) if h['weight_decay'] > 0 else -6 for h in hparams_list])
dropout_scatter = [int(h['dropout_input'] == 0.2) for h in hparams_list]
colors = []
for i in range(len(hparams_list)):
    cur_hparam_tuple = utils.hparamDictToTuple(hparams_list[i])
    colors.append(results_no_distill[cur_hparam_tuple]['train_acc'][-1])

marker_size = 100
fig, ax = plt.subplots()
plt.scatter(weight_decay_scatter, dropout_scatter, marker_size, c=colors, edgecolors='black')
plt.colorbar()
for i in range(len(weight_decay_scatter)):
    ax.annotate(str('%0.4f' % (colors[i],)), (weight_decay_scatter[i], dropout_scatter[i]))
plt.show()

num_epochs = 60
print_every = 100

temperatures = [10]
# trade-off between soft-target (st) cross-entropy and true-target (tt) cross-entropy;
# loss = alpha * st + (1 - alpha) * tt
alphas = [0.5]
learning_rates = [1e-2]
learning_rate_decays = [0.95]
weight_decays = [1e-5]
momentums = [0.9]
dropout_probabilities = [(0.0, 0.0)]
hparams_list = []
for hparam_tuple in itertools.product(alphas, temperatures, dropout_probabilities, weight_decays, learning_rate_decays,
                                        momentums, learning_rates):
    hparam = {}
    hparam['alpha'] = hparam_tuple[0]
    hparam['T'] = hparam_tuple[1]
    hparam['dropout_input'] = hparam_tuple[2][0]
    hparam['dropout_hidden'] = hparam_tuple[2][1]
    hparam['weight_decay'] = hparam_tuple[3]
    hparam['lr_decay'] = hparam_tuple[4]
    hparam['momentum'] = hparam_tuple[5]
    hparam['lr'] = hparam_tuple[6]
    hparams_list.append(hparam)

results_distill = {}
for hparam in hparams_list:
    print('Training with hparams' + utils.hparamToString(hparam))
    reproducibilitySeed()
    student_net = networks.StudentNetwork()
    student_net = student_net.to(fast_device)
    hparam_tuple = utils.hparamDictToTuple(hparam)
    results_distill[hparam_tuple] = utils.trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs,
                                                                train_val_loader, None,
                                                                print_every=print_every,
                                                                fast_device=fast_device)
    save_path = checkpoints_path_student + utils.hparamToString(hparam) + '_final.tar'
    torch.save({'results' : results_distill[hparam_tuple],
                'model_state_dict' : student_net.state_dict(),
                'epoch' : num_epochs}, save_path)

# Calculate student test accuracy
_, test_accuracy = utils.getLossAccuracyOnDataset(student_net, test_loader, fast_device)
print('student test accuracy (w distillation): ', test_accuracy)