import os
import random
import time

import torch
import numpy as np

from models.alt_model.model import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, FocalLoss, load_prior
from metrics import evaluate_codes, evaluate_hf
import argparse
import torch.nn as nn
import torch.nn.functional as F

def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--seed', help='Random seed', type=int, default=42)
    parser.add_argument('--dataset', help='name of dataset', type=str, default='mimic3') #mimic3 or mimic4
    parser.add_argument('--task', help='name of prediction', type=str, default='h') #diabetes or h
    parser.add_argument('--code_embedding_size', help='size of code embedding', type=int, default=64) 
    parser.add_argument('--graph_size', help='hidden size of graph', type=int, default=64) 
    parser.add_argument('--batch_size', help='batch size', type=int, default=4) 
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=10) 
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001) 
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=1e-6) 
    parser.add_argument('--dropout', help='dropout', type=float, default=0.2) 
    parser.add_argument('--loss_fn', help='type of loss function', type=str, default='focal') 
    parser.add_argument('--result_save_path', help='path to save the test results', type=str, default='log/mimic3/diabetes/chet/') 
    parser.add_argument('--resume_training', help='resume training from previous checkpoint or not',  action="store_true", default=False)
    parser.add_argument('--eval_steps', help='Number of step per eval',  type=int, default=250)
    parser.add_argument('--early_stopping', help='Number of rounds w/o improvements to early stop',  type=int, default=8)

    args = parser.parse_args()

    return args


import logging
import datetime

from datetime import datetime

# datetime object containing current date and time

# logging.warning('This will get logged to a file')
class CustomBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1):
      super().__init__()
      self.pos_weight = pos_weight

    def forward(self, input, target):
      epsilon = 10 ** -44
      input = input.sigmoid().clamp(epsilon, 1 - epsilon)

      my_bce_loss = -1 * (self.pos_weight * target * torch.log(input)
                          + (1 - target) * torch.log(1 - input))
      add_loss = (target - 0.5) ** 2 * 4
      mean_loss = (my_bce_loss * add_loss).mean()
      return mean_loss


class CustomKLDivLoss(torch.nn.Module):
    def __init__(self, temperature = 5):
      super().__init__()
      self.temperature = temperature

    def forward(self, teacher_outputs, outputs):
        T = self.temperature
        # print(teacher_outputs.sum(), outputs.sum())
        # print(T)
        # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        #                         F.softmax(teacher_outputs/T, dim=1)) * (  T * T) 
        # print(KD_loss)
        return KD_loss


def loss_fn_kd(teacher_outputs, outputs, params = {'temperature':5}):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    """
    T = params.temperature
    # print(T)
    print(outputs, teacher_outputs)
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (  T * T) 
    # print(KD_loss)
    return KD_loss

if __name__ == '__main__':

    args = read_option()
    seed = args.seed
    dataset = args.dataset
    task = args.task
    
    code_size = args.code_embedding_size
    graph_size =args.graph_size
    hidden_size = 150
    t_attention_size = 32
    t_output_size =1
    batch_size = args.batch_size
    epochs = args.epochs
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    num_early_stopping_rounds = args.early_stopping

    eval_steps = args.eval_steps
    # seed = 42
    # dataset = 'mimic4'  # 'mimic3' or 'eicu'
    # # task = 'diabetes'  # 'm' or 'h'
    # task = 'diabetes'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    result_save_path = args.result_save_path
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)
    print('code_num',code_num)


    # if task == 'h':
    #     prior = load_prior(f'data/{dataset}/standard/hf_prior.npz', device=device)
    # elif task == 'diabetes':
    #     prior = load_prior(f'data/{dataset}/standard/diabetes_prior.npz', device=device)
    # prior = prior.transpose()
    # print(prior.shape)
    # prior = np.mean(prior,axis=0)
    # prior = prior.mean(axis=1)
    # prior = prior.mean(axis = 1)

    # print(prior.grad)
    # prior = prior / prior.sum()
    # print(prior.sum())

    # m = prior.mean(0, keepdim=True)
    # s = prior.std(0, unbiased=False, keepdim=True)
    # prior -= m
    # prior /= s    # print(torch.nonzero(prior), prior.shape)
    # print(prior)
    # # prior = prior/prior.sum()
    # # prior = prior.unsqueeze(0).repeat(batch_size,1)
    # print(prior, prior.shape)
    # prior_min, prior_max = prior.min(), prior.max()
    # new_min, new_max = 0.0, 1.0
    # prior = (prior - prior_min)/(prior_max - prior_min)*(new_max - new_min) + new_min
    # print(prior, prior.shape)
    # prior = torch.Tensor([1.0 for i in range(code_num)]).to(device=device, dtype=torch.float32)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

    print(len(train_data))

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.2,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.001,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        },

        'diabetes': {
            'dropout': 0.2,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.001,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }

    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()

    # prior_loss = CustomBCELoss( )
    # loss_fn = torch.nn.BCELoss()
    prior_loss = torch.nn.MSELoss()
    # prior_loss = CustomKLDivLoss()
    loss_fn = FocalLoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = args.dropout

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=150,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
    #                                  task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    # scheduler = 
    # scheduler = 

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=lr,cycle_momentum= False, step_size_up = 80)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.25,patience=3,verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    # if args.resume_training:
    #     model.load_state_dict(torch.load(os.path.join(param_path, 'best_bert.pt')))
    

    param_path = 'params_alt/'
    if not os.path.exists(param_path):
        os.makedirs(param_path)
    best_val_f1_auc = -999
    model_save_path = os.path.join(param_path, f'{task}_{dataset}.pt')

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = dt_string.replace('/','_')
    dt_string = dt_string.replace(':','_')

    log_file_path = os.path.join(param_path, dt_string + f' {task}_{dataset}.log')
    # print(dt_string)

    logging.basicConfig(filename=log_file_path, filemode='w', level=logging.INFO)
    logging.info(f'Training dataset {dataset} on task {task}')

    early_stopping_count = 0 
    early_stopping_bool= False
    for epoch in range(epochs):
        if early_stopping_bool:
            break
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        if epoch == 1:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        elif epoch == 2:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        elif epoch >= 2:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.25

        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        batch_index = 0
        avg_prior_loss = []
        for step in range(len(train_data)):
            model.train()
            if early_stopping_bool:
                break

            optimizer.zero_grad()

            code_x, visit_lens, divided, y, neighbors = train_data[step]
            output,attention_weights = model(code_x, divided, neighbors, visit_lens, prior, return_attention  = True)
            output  = output.squeeze()
            prior_regularize_loss = 0
            # print(attention_weights.shape)
            prior_vector = prior.unsqueeze(0).repeat(attention_weights.shape[0],1)
            # prior_vector = prior_vector[code_x.nonzero(as_tuple = True)]

            # prior_vector = prior_vector[code_x.nonzero(as_tuple = True)]
            # attention_weights /= attention_weights.sum()
            # attention_weights = attention_weights[code_x.nonzero(as_tuple = True)]

            # prior_vector /= prior_vector.sum()
            # print(prior_vector.shape)
            # print(code_x.shape)
            non_zero_code_indices = []
            for i in range(len(code_x)):
                visit_len_i = visit_lens[i]
                for visit_indx in range(visit_len_i):
                    non_zero_indices = code_x[i,visit_indx,:].nonzero().squeeze()
                    non_zero_code_indices.append(non_zero_indices)
                    # prior_vector_present = prior_vector[]
                    # print(non_zero_indices.shape)
                # print(non_zero_indices)
            prior_regularize_loss = 0
            for i in range(len(non_zero_code_indices)):
                prior_vector_i , attention_weights_i, non_zero_i = prior_vector[i], attention_weights[i],\
                                                                    non_zero_code_indices[i]
                # print(prior_vector_i.shape)
                # if prior_vector_i_filtered.shape[0] != 0:
                try:
                    prior_vector_i_filtered = prior_vector_i[non_zero_i]
                    attention_weights_i_filtered = attention_weights_i[non_zero_i]
                    # print(prior_vector_i_filtered.shape, attention_weights_i_filtered.shape)
                    # print(attention_weights_i_filtered)
                    if prior_vector_i_filtered.sum() > 1e-12:
                        prior_vector_i_filtered /= prior_vector_i_filtered.sum()
                        # print("Before",prior_vector_i_filtered.sum())
                        prior_vector_i_filtered  = (prior_vector_i_filtered + 1)/ (prior_vector_i_filtered + 1).sum()
                        # print("After",prior_vector_i_filtered.sum())

                    else:
                        prior_vector_i_filtered = torch.Tensor([1/len(prior_vector_i_filtered) for i in range(len(prior_vector_i_filtered))]).to(device)
                    # print(attention_weights_i_filtered)
                    # prior_regularize_loss += prior_loss(prior_vector_i_filtered  , attention_weights_i_filtered)
                    prior_regularize_loss += prior_loss(prior_vector_i_filtered  , attention_weights_i_filtered)

                except:
                    pass
            # for i in range(len(attention_weights)):
            #     # print(visit_lens[i])
            #     visit_len = visit_lens[i].item()
            #     code_x_index = code_x[i][:visit_len,:].long()
            #     attention = attention_weights[i]
            #     # print(attention)
            #     # print(attention.shape, prior.shape)
            #     prior_vector = prior.unsqueeze(0).repeat(attention.shape[0],1)
            #     # prior_vector = prior_vector * code_x_index
            #     prior_vector = prior_vector[code_x_index.nonzero(as_tuple = True)]
            #     # print(attention[code_x_index.nonzero(as_tuple = True)])

            #     # attention = attention * code_x_index
            #     attention = attention[code_x_index.nonzero(as_tuple = True)]
            #     # print('prior',prior_vector)
            #     if prior_vector.sum() > 0:
            #         prior_vector /= prior_vector.sum()
            #     # else:
            #     #     prior_vector = prior_vec
            #     attention /= attention.sum()
            #     print('prior',prior_vector)
            #     print('attention',attention)
            #     prior_regularize_loss += prior_loss(prior_vector + 1e-9 , attention)
            prior_regularize_loss = prior_regularize_loss / len(non_zero_code_indices)
            # avg_prior_loss.append(prior_regularize_loss.item())
            avg_prior_loss.append(prior_regularize_loss)

            if step % 50 == 0:
                print('')
                print(sum(avg_prior_loss) / len(avg_prior_loss))

            loss = loss_fn(output, y)
            # print(loss)
            loss = loss + 0.05  * prior_regularize_loss
            loss.backward()
            # print(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
            % (step + 1, steps, remaining_time, total_loss / total_num), end='')
            # logging.info('\r    Step %d / %d, remaining time: %s, loss: %.4f'
            # % (step + 1, steps, remaining_time, total_loss / total_num))
            if (batch_index +1) % eval_steps == 0:
                # print(batch_index)
                valid_loss, auc,f1_score,report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical, prior)
                auc_f1_avg = (f1_score + auc) / 2 
                scheduler.step(auc_f1_avg)
                
                if auc_f1_avg > best_val_f1_auc:
                    torch.save(model.state_dict(), model_save_path)
                    print('Saved checkpoint')
                    best_val_f1_auc = auc_f1_avg
                    early_stopping_count = 0
                else:
                    early_stopping_count +=1
                    if early_stopping_count >= num_early_stopping_rounds:
                        early_stopping_bool = True

                    print(f"Early stopping count: {early_stopping_count}")
            batch_index += 1

        # if early_stopping_count >= num_early_stopping_rounds:
        #     break
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, auc,f1_score,report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical,prior)
        auc_f1_avg = (f1_score + auc) / 2 
        scheduler.step(auc_f1_avg)

        if auc_f1_avg > best_val_f1_auc:
            torch.save(model.state_dict(),model_save_path)
            print('Saved checkpoint')
            best_val_f1_auc = auc_f1_avg
    # torch.save('')
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    logging.info('Evaluating on the validation set')
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    _,val_auc,val_f1,val_report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical, prior)
    # print(val_auc, val_f1, val_report)
    logging.info(f'Validation auc: {val_auc} - Validation F1: {val_f1}')
    logging.info(val_report)

    logging.info('Evaluating on the test set')
    _,test_f1,test_auc,test_report = evaluate_fn(model, test_data, loss_fn, output_size, test_historical, prior)
    # print(test_auc,test_f1,test_report)
    logging.info(f'Test auc: {test_auc} - Test F1: {test_f1}')

    logging.info(test_report)
        # torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
