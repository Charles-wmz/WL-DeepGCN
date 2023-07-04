from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from models import GCN

from metrics import torchmetrics_accuracy, torchmetrics_auc, correct_num, prf

from dataloader import dataloader

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=46, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) default: ho, see preprocessed-connectomes-project.org/abide/Pipelines.html for more options ')
parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for the feature selection step (default: 2000)')
parser.add_argument('--folds', default=10, type=int, help='For cross validation, specifies which fold will be used. All folds are used if set to 11 (default: 11)')
parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network construction (default: correlation, options: correlation, partial correlation, tangent)')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--ngl', default=8, type=int, help='number of gcn hidden layders (default: 8)')
parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
parser.add_argument('--train', default=1, type=int, help='train(default: 1) or evaluate(0)')
parser.add_argument('--ckpt_path', type=str, default='./pth', help='checkpoint path to save trained models')
parser.add_argument('--early_stopping', action='store_true', default=True, help='early stopping switch')
parser.add_argument('--early_stopping_patience', type=int, default=20, help='early stoppng epochs')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
params = dict()
params['no_cuda'] = args.no_cuda
params['seed'] = args.seed
params['epochs'] = args.epochs
params['lr'] = args.lr
params['weight_decay'] = args.weight_decay
params['hidden'] = args.hidden
params['dropout'] = args.dropout
params['atlas'] = args.atlas
params['num_features'] = args.num_features
params['folds'] = args.folds
params['connectivity'] = args.connectivity
params['max_degree'] = args.max_degree
params['ngl'] = args.ngl
params['edropout'] = args.edropout
params['train'] = args.train
params['ckpt_path'] = args.ckpt_path
params['early_stopping'] = args.early_stopping
params['early_stopping_patience'] = args.early_stopping_patience

# Print Hyperparameters
print('Hyperparameters:')
for key, value in params.items():
    print(key + ":", value)

corrects = np.zeros(args.folds, dtype=np.int32) 
accs = np.zeros(args.folds, dtype=np.float32) 
aucs = np.zeros(args.folds, dtype=np.float32)
prfs = np.zeros([args.folds,3], dtype=np.float32) # Save Precision, Recall, F1
test_num = np.zeros(args.folds, dtype=np.float32)

print('  Loading dataset ...')
dataloader = dataloader()
raw_features, y, nonimg = dataloader.load_data(params) # 数据加载：原始特征，标签，非成像数据
cv_splits = dataloader.data_split(args.folds)

t1 = time.time()

for i in range(args.folds):
    t_start = time.time()
    train_ind, test_ind = cv_splits[i]

    train_ind, valid_ind = train_test_split(train_ind, test_size=0.1, random_state = 24)
    
    cv_splits[i] = (train_ind, valid_ind)
    cv_splits[i] = cv_splits[i] + (test_ind,)
    print('Size of the {}-fold Training, Validation, and Test Sets:{},{},{}' .format(i+1, len(cv_splits[i][0]), len(cv_splits[i][1]), len(cv_splits[i][2])))

    if args.train == 1:
        for j in range(args.folds):
            print(' Starting the {}-{} Fold:：'.format(i+1,j+1))
            node_ftr = dataloader.get_node_features(train_ind)
            edge_index, edgenet_input = dataloader.get_WL_inputs(nonimg)
            edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
            
            model = GCN(input_dim = args.num_features,
                        nhid = args.hidden, 
                        num_classes = 2, 
                        ngl = args.ngl, 
                        dropout = args.dropout, 
                        edge_dropout = args.edropout, 
                        edgenet_input_dim = 2*nonimg.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.cuda:
                model.cuda()
                features = torch.tensor(node_ftr, dtype=torch.float32).cuda()
                edge_index = torch.tensor(edge_index, dtype=torch.long).cuda()
                edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).cuda()
                labels = torch.tensor(y, dtype=torch.long).cuda()
                fold_model_path = args.ckpt_path + "/fold{}.pth".format(i+1)
                
            acc = 0
            best_val_loss = float('inf') # early stoppping: Initialized to positive infinity
            current_patience = 0 # early stopping: Used to record the epochs of the current early stopping

            for epoch in range(args.epochs):
                # train
                model.train()
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    output, edge_weights = model(features, edge_index, edgenet_input)
                    loss_train = torch.nn.CrossEntropyLoss()(output[train_ind], labels[train_ind])
                    loss_train.backward()
                    optimizer.step()
                acc_train = torchmetrics_accuracy(output[train_ind], labels[train_ind])
                auc_train = torchmetrics_auc(output[train_ind], labels[train_ind])
                logits_train = output[train_ind].detach().cpu().numpy()
                prf_train = prf(logits_train, y[train_ind])

                
                # valid
                model.eval()
                with torch.set_grad_enabled(False):
                    output, edge_weights = model(features, edge_index, edgenet_input)
                loss_val = torch.nn.CrossEntropyLoss()(output[valid_ind], labels[valid_ind])
                acc_val = torchmetrics_accuracy(output[valid_ind], labels[valid_ind])
                auc_val = torchmetrics_auc(output[valid_ind], labels[valid_ind])
                logits_val = output[valid_ind].detach().cpu().numpy()
                prf_val = prf(logits_val, y[valid_ind])

                
                print('Epoch:{:04d}'.format(epoch+1))
                print('acc_train:{:.4f}'.format(acc_train),
                      'pre_train:{:.4f}'.format(prf_train[0]),
                      'recall_train:{:.4f}'.format(prf_train[1]),
                      'F1_train:{:.4f}'.format(prf_train[2]),
                      'AUC_train:{:.4f}'.format(auc_train))
                print('acc_val:{:.4f}'.format(acc_val),
                      'pre_val:{:.4f}'.format(prf_val[0]),
                      'recall_val:{:.4f}'.format(prf_val[1]),
                      'F1_val:{:4f}'.format(prf_val[2]),
                      'AUC_val:{:.4f}'.format(auc_val))
                
                # save pth
                if acc_val > acc and epoch > 50:
                    acc = acc_val
                    if args.ckpt_path != '':
                        if not os.path.exists(args.ckpt_path):
                            os.makedirs(args.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)
                
                # Early Stopping
                if epoch > 50 and args.early_stopping == True:
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        current_patience = 0
                    else:
                        current_patience += 1
                    if current_patience >= args.early_stopping_patience:
                        print('Early Stopping!!! epoch：{}'.format(epoch))
                        break
                        
        # test
        print("Loading the Model for the {}-th Fold:... ...".format(i+1),
              "Size of samples in the test set:{}".format(len(test_ind)))
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()
        
        with torch.set_grad_enabled(False):
            output, edge_weights = model(features, edge_index, edgenet_input)
        acc_test = torchmetrics_accuracy(output[test_ind], labels[test_ind])
        auc_test = torchmetrics_auc(output[test_ind], labels[test_ind])
        logits_test = output[test_ind].detach().cpu().numpy()
        correct_test = correct_num(logits_test, y[test_ind])
        prf_test =  prf(logits_test, y[test_ind])
        
        t_end = time.time()
        t = t_end - t_start
        print('Fold {} Results:'.format(i+1),
              'test acc:{:.4f}'.format(acc_test),
              'test_pre:{:.4f}'.format(prf_test[0]),
              'test_recall:{:.4f}'.format(prf_test[1]),
              'test_F1:{:.4f}'.format(prf_test[2]),
              'test_AUC:{:.4f}'.format(auc_test),
              'time:{:.3f}s'.format(t))
        
        correct = correct_test
        aucs[i] = auc_test
        prfs[i] = prf_test
        corrects[i] = correct
        test_num[i] = len(test_ind)
    
    
    if args.train == 0:
        node_ftr = dataloader.get_node_features(train_ind)
        edge_index, edgenet_input = dataloader.get_WL_inputs(nonimg)
        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        
        model = GCN(input_dim = args.num_features,
                    nhid = args.hidden, 
                    num_classes = 2, 
                    ngl = args.ngl, 
                    dropout = args.dropout, 
                    edge_dropout = args.edropout, 
                    edgenet_input_dim = 2*nonimg.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.cuda:
            model.cuda()
            features = torch.tensor(node_ftr, dtype=torch.float32).cuda()
            edge_index = torch.tensor(edge_index, dtype=torch.long).cuda()
            edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).cuda()
            labels = torch.tensor(y, dtype=torch.long).cuda()
            fold_model_path = args.ckpt_path + "/fold{}.pth".format(i+1)
        
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()
        
        with torch.set_grad_enabled(False):
            output, edge_weights = model(features, edge_index, edgenet_input)
        acc_test = torchmetrics_accuracy(output[test_ind], labels[test_ind])
        auc_test = torchmetrics_auc(output[test_ind], labels[test_ind])
        logits_test = output[test_ind].detach().cpu().numpy()
        correct_test = correct_num(logits_test, y[test_ind])
        prf_test =  prf(logits_test, y[test_ind])
        
        t_end = time.time()
        t = t_end - t_start
        print('Fold {} Results:'.format(i+1),
              'test acc:{:.4f}'.format(acc_test),
              'test_pre:{:.4f}'.format(prf_test[0]),
              'test_recall:{:.4f}'.format(prf_test[1]),
              'test_F1:{:.4f}'.format(prf_test[2]),
              'test_AUC:{:.4f}'.format(auc_test),
              'time:{:.3f}s'.format(t))
        
        correct = correct_test
        aucs[i] = auc_test
        prfs[i] = prf_test
        corrects[i] = correct
        test_num[i] = len(test_ind)

t2 = time.time()

print('\r\n======Finish Results for Nested 10-fold cross-validation======')
Nested10kCV_acc = np.sum(corrects) / np.sum(test_num)
Nested10kCV_auc = np.mean(aucs)
Nested10kCV_precision, Nested10kCV_recall, Nested10kCV_F1 = np.mean(prfs, axis=0)
print('Test:',
      'acc:{}'.format(Nested10kCV_acc),
      'precision:{}'.format(Nested10kCV_precision),
      'recall:{}'.format(Nested10kCV_recall),
      'F1:{}'.format(Nested10kCV_F1),
      'AUC:{}'.format(Nested10kCV_auc))
print('Total duration:{}'.format(t2 - t1))

