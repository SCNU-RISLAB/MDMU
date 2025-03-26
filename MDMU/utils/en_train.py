import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from MainModel import  Model


import random
import numpy as np
from utils.data_loader import data_loader
from itertools import chain
import time

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from thop import profile

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


class EnConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M':1,
                    # 'T':1,
                    # 'A':1,
                    # 'V':1,
                },
                 model_save_path = 'checkpoint/',
                 learning_rate = 5e-6,
                 epochs = 20,
                 dataset_name = 'MOSI',
                 early_stop = 5,
                 seed = 0,
                 dropout=0.3,
                 batch_size = 16,
                 multi_task = False,
                 tasks = 'M',
                 context = False,
                ):

        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.video_context_len = video_context_len

        
        
class EnTrainer():
    def __init__(self, config):
 
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        print(self.criterion)
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader, first_epoch:bool):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)

        total_loss = 0
        # Loop over all batches.
        all_embeddings = []
        labels = []
        for i , batch in tqdm(enumerate(data_loader)):
            video_inputs = batch["video_features"].to(device)
            video_context_inputs = batch["video_context_features"].to(device)

            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)

            audio_inputs = batch["audio_inputs"].to(device)
            # audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            # audio_context_mask = batch["audio_context_masks"].to(device)

            targets = batch["targets"].to(device).view(-1, 1)

            optimizer.zero_grad()                    # To zero out the gradients.

            if self.config.context:
                outputs = model(video_inputs, video_context_inputs, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_context_inputs)
            else:
                outputs = model(video_inputs, text_inputs, text_mask, audio_inputs)


            # Compute the training loss.
            if self.config.multi_task:
                loss = 0.0         
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
    #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss_L1 = self.criterion(outputs['M'], targets)
                loss_CL = outputs['CL_loss']
                loss = loss_L1 + loss_CL
                total_loss += loss.item()*text_inputs.size(0)
        
            loss.backward()
            optimizer.step()                
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)

        if first_epoch:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(labels, dim=0)


        return total_loss

    def do_test(self, model, data_loader, mode):
        print("start testing")
        model.eval()   # Put the model in eval mode.
        if self.config.multi_task:
            # y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            # y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            y_pred = {'M': []}
            y_true = {'M': []}
            total_loss = 0
            val_loss = {
                'M':0,
                # 'T':0,
                # 'A':0,
                # 'V':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            all_embeddings=[]
            labels7=[]
            labels2=[]

            for i, batch in tqdm(enumerate(data_loader)):                    # Loop over all batches.
                video_inputs = batch["video_features"].to(device)
                video_context_inputs = batch["video_context_features"].to(device)

                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                starter.record()
                if self.config.context:
                    outputs = model(video_inputs, video_context_inputs, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_context_inputs)
                else:
                    outputs = model(video_inputs, text_inputs, text_mask, audio_inputs, audio_mask)
                ender.record()
                torch.cuda.synchronize()
                latency = starter.elapsed_time(ender)
                print(f"Latency: {latency:.3f} ms")
                times.append(latency)

                if mode == 'TEST':
                    targets7 = torch.round(targets).cpu()
                    labels7.append(targets7)
                    targets2 = torch.sign(targets)
                    targets2[targets2 == 0] = 1
                    targets2 = targets2.cpu()
                    labels2.append(targets2)

                    embedding = outputs['embedding']
                    all_embeddings.append(embedding)


                # Compute loss.
                if self.config.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

            avg_latency = sum(times)/len(times)
            print(f"Average Latency: {avg_latency:.3f} ms")
            # sys.exit("Terminating the program at this point.")

            if mode == 'TEST':
                all_embeddings = torch.cat(all_embeddings, dim=0)
                all_labels7 = torch.cat(labels7, dim=0)
                all_labels2 = torch.cat(labels2, dim=0)
                # np.save('all_embeddings_2-6.npy', all_embeddings.detach().numpy())
                # np.save('all_labels7_2-6.npy', all_labels7.detach().numpy())
                # np.save('all_labels2_2-6.npy', all_labels2.detach().numpy())




        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss

        return eval_results


def EnRun(config):
    # torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len,
                                                        video_context_length=config.video_context_len,
                                                        seed=config.seed,)

    # print(config.context)
    print(config.model)
    if config.context:
        model = Model(config).to(device)

    else:
        print("fuck off")


    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    highest_eval_acc7 = 0
    epoch = 0
    best_epoch = 0
    first_epoch = True
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        if epoch!=0:
            first_epoch=False
        trainer.do_train(model, train_loader, first_epoch)
        eval_results = trainer.do_test(model, test_loader,"VAL")
        torch.save(model.state_dict(), config.model_save_path + 'last_model.pth')
        test_results = trainer.do_test(model, test_loader,"TEST")
        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+'RH_loss.pth')
            best_epoch = epoch
        if eval_results['Has0_acc_2']>highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            torch.save(model.state_dict(), config.model_save_path+'RH_acc.pth')
        if eval_results['Mult_acc_7']>highest_eval_acc7:
            highest_eval_acc7 = eval_results['Mult_acc_7']
            torch.save(model.state_dict(), config.model_save_path+'RH_acc7.pth')
            # best_epoch = epoch
        if epoch - best_epoch >= config.early_stop:
            break
        print(f"\n current highest acc7：{highest_eval_acc7}")
        print(f" current highest acc2：{highest_eval_acc}")
        print(f" remaining patience ：{config.early_stop - epoch + best_epoch}")
        print(f" current lowest loss epoch：{best_epoch}")

        epoch += 1

    model.load_state_dict(torch.load(config.model_save_path+'RH_acc.pth'))
    test_results_loss = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path+'RH_loss.pth'))
    test_results_acc = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))

    model.load_state_dict(torch.load(config.model_save_path + 'RH_acc7.pth'))
    test_results_acc7 = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (highest val acc7) ') + dict_to_str(test_results_acc7))

    output_file = config.model_save_path + 'test_results.txt'

    with open(output_file, 'w') as f:
        f.write('%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_loss) + '\n')
        f.write('%s: >> ' % ('TEST (lowest val loss) ') + dict_to_str(test_results_acc) + '\n')
        f.write('%s: >> ' % ('TEST (highest val acc7) ') + dict_to_str(test_results_acc7) + '\n')
        
