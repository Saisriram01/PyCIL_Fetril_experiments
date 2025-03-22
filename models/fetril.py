'''

results on CIFAR-100: 
               
           |   Reported  Resnet18        |  Reproduced Resnet32 
Protocols  |  Reported FC | Reported SVM |  Reproduced FC | Reproduced SVM |  

T = 5      |   64.7       |  66.3        |  65.775        | 65.375         |

T = 10     |   63.4       |  65.2        |  64.91         | 65.10          |

T = 60     |   50.8       |  59.8        |  62.09         | 61.72          |

'''


import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from sklearn.svm import LinearSVC
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy,ImageNetPolicy
from utils.ops import Cutout
from convs.linears import SimpleLinear

EPSILON = 1e-8


def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss


    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")



class FeTrIL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._means = []
        self._svm_accs = []

    def after_task(self):
        self._known_classes = self._total_classes
        
    def incremental_train(self, data_manager):
        print("Calling incremental_train")
        self.data_manager = data_manager
        self.data_manager._train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        ]
        self._cur_task += 1

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)


        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            for p in self._network.convnet.parameters():
                p.requires_grad = False
        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            self._epoch_num = self.args["init_epochs"]
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
            self._compute_means()
            self._build_feature_set()
        else:
            self._epoch_num = self.args["epochs"]
            self._compute_means()
            self._compute_relations()
            self._build_feature_set()

            train_loader = DataLoader(self._feature_trainset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
            optimizer = optim.SGD(self._network_module_ptr.fc.parameters(),momentum=0.9,lr=self.args["lr"],weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = self.args["epochs"])
            
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._train_svm(self._feature_trainset,self._feature_testset)

    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)        
            
    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations=np.argmax((old_means/np.linalg.norm(old_means,axis=1)[:,None])@(new_means/np.linalg.norm(new_means,axis=1)[:,None]).T,axis=1)+self._known_classes

    def _build_feature_set(self):
        self.vectors_train = []
        self.labels_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx]*len(vectors))
        for class_idx in range(0,self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(self.vectors_train[new_idx-self._known_classes]-self._means[new_idx]+self._means[class_idx])
            self.labels_train.append([class_idx]*len(self.vectors_train[-1]))
        
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train,self.labels_train)
        
        self.vectors_test = []
        self.labels_test = []
        for class_idx in range(0, self._total_classes):                     # change to generate features only for new classes and see if accuracy changes
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='test',
                                                                mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            self.vectors_test.append(vectors)
            self.labels_test.append([class_idx]*len(vectors))
        self.vectors_test = np.concatenate(self.vectors_test)
        self.labels_test = np.concatenate(self.labels_test)
        self._feature_testset = FeatureDataset(self.vectors_test,self.labels_test)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):  #Using Separated Softmax

        TEMP = 2 # Temperature for softmax
        prog_bar = tqdm(range(self._epoch_num))
        loss_KD = 0
        
        task_last_class = [0, self.data_manager.get_task_size(0)]
        for i in range(1,self._cur_task+1):
            task_last_class.append(self.data_manager.get_task_size(i) + task_last_class[-1])

        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task ==0:
                    logits = self._network(inputs)['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']
                    score = self._old_fc(inputs)['logits'].data
                    loss_KD = torch.zeros(self._cur_task).cuda()
                    for t in range(1,self._cur_task+1):

                        start_KD = task_last_class[t-1]
                        end_KD = task_last_class[t]

                        soft_target = F.softmax(score[:,start_KD:end_KD] / TEMP, dim=1)
                        output_log = F.log_softmax(logits[:,start_KD:end_KD] / TEMP, dim=1)
                        #print(f"output_log, soft_target = {output_log.shape, soft_target.shape}")
                        loss_KD[t-1] = F.kl_div(output_log, soft_target, reduction='batchmean') * (TEMP**2)

                    loss_KD = loss_KD.sum()
                loss = F.cross_entropy(logits, targets) + loss_KD                                    # loss function -------------------------------------------------

                #loss = nca(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, SSIL_KD {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), loss_KD, train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, SSIL_KD {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), loss_KD, train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

        self._old_fc = self._network_module_ptr.fc

    def _train_svm(self,train_set,test_set):
        print("Training the SVM classifier now!\n\n")
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        test_features = test_set.features.numpy()
        test_labels = test_set.labels.numpy()
        train_features = train_features/np.linalg.norm(train_features,axis=1)[:,None]
        test_features = test_features/np.linalg.norm(test_features,axis=1)[:,None]
        svm_classifier = LinearSVC(random_state=42)
        svm_classifier.fit(train_features,train_labels)
        logging.info("svm train: acc: {}".format(np.around(svm_classifier.score(train_features,train_labels)*100,decimals=2)))
        acc = svm_classifier.score(test_features,test_labels)
        self._svm_accs.append(np.around(acc*100,decimals=2))
        logging.info("svm evaluation: acc_list: {}".format(self._svm_accs))

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label
