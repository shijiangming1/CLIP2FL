import os
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num, Clip_Indices2Dataset
from Dataset.sample_dirichlet import clients_indices
from Dataset.Gradient_matching_loss import match_loss
import numpy as np
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8_256 import ResNet_cifar

# add
# from Model_teacher import model_dict
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.nn.functional as F
import sys
import pickle
import clip
from PIL import Image
from losses import SupConLoss_text

from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
import time
from Dataset.param_aug import DiffAugment
import datetime
import shutil

def load_labels_name(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_clip_input(images, preprocess, transform_train):
    """
        trans tensor 32 * 32 to get clip_input 224 * 224 tensor
    """

    images_copy = images.cpu().clone()
    bc = images.shape[0]

    image_input = preprocess(transform_invert(images_copy[0], transform_train)).unsqueeze(0).to(args.device)

    for i in range(1, bc):
        PIL_image = transform_invert(images_copy[i], transform_train)
        tmp_tensor = preprocess(PIL_image).unsqueeze(0).to(args.device)  # torch.Size([1, 3, 224, 224])
        image_input = torch.cat([image_input, tmp_tensor], 0)

    return image_input


def transform_invert(img_, transform_train):
    """
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_

def load_data_cifar(filename, mode='cifar10'):
    """ load data and labels information from cifar10 and cifar100
    cifar10 keys(): dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    cifar100 keys(): dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
    """
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        if mode == 'cifar10':
            data = dataset[b'data']
            labels = dataset[b'labels']
            img_names = dataset[b'filenames']
        elif mode == 'cifar100':
            data = dataset[b'data']
            labels = dataset[b'fine_labels']
            img_names = dataset[b'filenames']
        else:
            print("mode should be in ['cifar10', 'cifar100']")
            return None, None, None

    return data, labels, img_names

class Logger(object):
    logfile = ""

    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        return

    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, "a")
                self.log.write(message)
                self.log.close()
            except:
                pass

    def flush(self):
        pass

class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        # kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
        #               F.softmax(out_t / self.T, dim=1),
        #               reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return kd_loss

class BKD2Loss(nn.Module):

    def __init__(self, T):
        super(BKD2Loss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t, weight_lamda):
        pred_t = F.softmax(out_t/self.T, dim=1)
        pred_t = pred_t * weight_lamda
        pred_t = pred_t / pred_t.sum(1)[:, None]

        kd = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        pred_t,
                        reduction='none').mean(dim=0)
        kd_loss = kd.sum() * self.T * self.T

        return kd_loss

class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 512), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature)  # optimizer_img for synthetic data
        self.criterion = CrossEntropyLoss().to(args.device)

        # PCL loss
        self.contras_criterion = SupConLoss_text(args.device, args.ins_temp, args.num_classes)

        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        self.feature_net = nn.Linear(512, args.num_classes).to(args.device)

    def update_feature_syn(self, args, global_params, list_clients_gradient, new_text_features):
        feature_net_params = self.feature_net.state_dict()
        for name_param in reversed(global_params):
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break
        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}
        # aggregate the real feature gradients
        for i in range(args.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        for ep in range(args.match_epoch):
            loss_feature = torch.tensor(0.0).to(args.device)
            for c in range(args.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, 512))
                    lab_syn = torch.ones((self.num_of_feature,), device=args.device, dtype=torch.long) * c
                    # print("test lab_syn: ", lab_syn, lab_syn.shape)
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)
            contrast_loss = self.contras_criterion(self.feature_syn, self.label_syn, new_text_features)
            # Eq. 8
            loss_feature += args.contrast_alpha * contrast_loss
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, fedavg_params, batch_size_local_training):
        feature_syn_train_ft = copy.deepcopy(self.feature_syn.detach())
        label_syn_train_ft = copy.deepcopy(self.label_syn.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(512, args.num_classes).to(args.device)
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        ft_model.train()
        for epoch in range(args.crt_epoch):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        feature_net_params = ft_model.state_dict()
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()

        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)
        self.kd_criterion = KDLoss(T=args.T).to(args.device)
        self.bkd2_criterion = BKD2Loss(T=args.T).to(args.device)
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def compute_gradient(self, global_params, args):
        # compute C^k
        list_class, per_class_compose = get_class_num(self.class_compose)  # class组成

        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        self.local_model.load_state_dict(global_params)

        self.local_model.eval()
        self.local_model.classifier.train()
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(args.device)
        # gradients of all classes
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        # choose to repeat 10 times
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                # transform
                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)

                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)
                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)

        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg

    def local_train(self, args, global_params, clip_model, text_features):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True, num_workers=1)
            for data_batch in data_loader:
                images, labels, clip_images = data_batch
                images, labels, clip_images = images.to(self.device), labels.to(self.device), clip_images.to(self.device)  # tensor
                images = transform_train(images)

                # compute client's output
                _, outputs = self.local_model(images)
                outputs = outputs.float()

                # get clip feature encode
                with torch.no_grad():
                    image_features = clip_model.encode_image(clip_images)
                image_features = image_features.float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_logits = (100.0 * image_features @ text_features.T)
                #Eq. 1
                loss1 = self.criterion(outputs, labels)
                loss2 = self.kd_criterion(outputs, clip_logits)
                loss = loss1 + args.alpha * loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()


def CLIP2FL():
    args = args_parser()
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
    random_state = np.random.RandomState(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # logger
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    log_name = current_time_str+'_main_clip2fl_' + str(args.alpha) + '_' + str(args.contrast_alpha) + '_' + 'IF' + str(args.imb_factor) + '.log'
    model_dir = os.path.join(args.result_save, args.dataset, 'main_clip2fl')
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    sys.stdout = Logger(os.path.join(model_dir, str(log_name)))
    sys.stderr = Logger(os.path.join(model_dir, str(log_name)))
    if not os.path.exists(args.result_save):
        os.mkdir(args.result_save)
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CLIP Loading
    clip_model, preprocess = clip.load('ViT-B/32', args.device)

    if args.num_classes == 10:
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        clip_data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=preprocess)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.num_classes == 100:
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
        clip_data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=preprocess)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)

    # get label_name from datasets
    if args.num_classes == 10:
        cifar10_path = "data/CIFAR10/cifar-10-batches-py"
        obj_cifar10 = load_labels_name(os.path.join(cifar10_path, 'batches.meta'))  
        label_name = obj_cifar10['label_names']
    elif args.num_classes == 100:
        cifar100_path = "data/CIFAR100/cifar-100-python"
        obj_cifar100 = load_labels_name(os.path.join(cifar100_path, 'meta'))  
        label_name = obj_cifar100['fine_label_names']

    # CLIP PART and Loading data
    clip_model.eval()
    text_inputs = clip.tokenize([f"a photo of a {c}" for c in label_name]).to(args.device) # torch.size([10, 77])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)

    text_features = text_features.float()
    text_features /= text_features.norm(dim=-1, keepdim=True) # torch.size([10, 512])

    new_text_features = text_features[0].repeat(100, 1)
    for i in range(1, args.num_classes):
        new_text_features = torch.cat([new_text_features, text_features[i].repeat(100,1)], 0)

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)

    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)

    # len(list_client2indices) = 20  [indices of each client]
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args,
                          num_of_feature=args.num_of_feature)
    total_clients = list(range(args.num_clients))
    indices2data = Clip_Indices2Dataset(data_local_training, clip_data_local_training)
    re_trained_acc = []
    temp_model = nn.Linear(512, args.num_classes).to(args.device)
    syn_params = temp_model.state_dict()

    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        syn_feature_params = copy.deepcopy(global_params)
        for name_param in reversed(syn_feature_params):
            if name_param == 'classifier.bias':
                syn_feature_params[name_param] = syn_params['bias']
            if name_param == 'classifier.weight':
                syn_feature_params[name_param] = syn_params['weight']
                break
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_clients_gradient = []
        list_dicts_local_params = []
        list_nums_local_data = []

        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data

            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            # compute the real feature gradients in local data
            truth_gradient = local_model.compute_gradient(copy.deepcopy(syn_feature_params), args)
            list_clients_gradient.append(copy.deepcopy(truth_gradient))
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params), clip_model, text_features)
            list_dicts_local_params.append(copy.deepcopy(local_params))
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params), list_clients_gradient, new_text_features)
        # re-trained classifier
        syn_params, ft_params = global_model.feature_re_train(copy.deepcopy(fedavg_params), args.batch_size_local_training)
        # global eval
        one_re_train_acc = global_model.global_eval(ft_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))

        if r % 10 == 0:
            print(re_trained_acc)

    print(re_trained_acc)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    CLIP2FL()


