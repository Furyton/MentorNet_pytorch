import os
from tqdm import tqdm
import utils
import torch
from . import reader
import datetime
import numpy as np
from abc import ABCMeta


# train_dir = ''
# data_path = ''
# device = 'cpu'
# mini_batch_size = 32
# max_step_train = 3e4
# learning_rate = 0.1
# worker_num = 2
# epoch = 2


class trainer(metaclass=ABCMeta):
    def __init__(self,  train_dir,
                        data_path,
                        device='cpu',
                        mini_batch_size=32,
                        learning_rate=0.1,
                        worker_num=2,
                        epoch=2,
                        show_progress_bar=False,
                        is_binary_label=True):

        
        self.train_dir = train_dir
        self.data_path = data_path
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.worker_num = worker_num
        self.epoch = epoch
        self.show_progress_bar = show_progress_bar
        self.is_binary_label = is_binary_label

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
        self.train_dataLoader = reader.get_train_dataloader(data_path=data_path,
        device=device,
        batch_size=mini_batch_size,
        worker_num=worker_num)
    
        self.test_dataLoader = reader.get_test_dataloader(data_path=data_path,
        device=device,
        batch_size=mini_batch_size,
        worker_num=worker_num)

        self.model = utils.MentorNet_nn()
        self.BCEloss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.MSEloss = torch.nn.MSELoss(reduction='mean')

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.lr_sheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim,gamma=0.9)


    def train(self):
        val_loss = self.test()

        for epoch in range(self.epoch):
            print("start training epoch: ", epoch)

            self.train_one_epoch(epoch)

            cur_loss = self.test()

            if cur_loss < val_loss:
                print(f'progress')
                val_loss = cur_loss
                self.save('best')

            self.lr_sheduler.step()
        
        self.save('final')


    def train_one_epoch(self, epoch):
        self.model.train()

        iterator = self.train_dataLoader if not self.show_progress_bar else tqdm(self.train_dataLoader)

        tot_loss = 0
        tot_batch = 0

        for batch_idx, batch in enumerate(iterator):
            self.optim.zero_grad()

            loss = self.calculate_loss(batch)

            tot_loss += loss.item()

            tot_batch += 1

            loss.backward()

            self.optim.step()

            if self.show_progress_bar:
                iterator.set_description('Epoch {}, loss {:.3f} '.format(epoch + 1, tot_loss / tot_batch))
            
        print(f'epoch: {epoch}, train loss: {tot_loss / tot_batch}')

    def test(self):
        self.model.eval()

        tot_loss = 0
        tot_batch = 0

        with torch.no_grad():
            iterator = self.test_dataLoader if not self.show_progress_bar else tqdm(self.test_dataLoader)

            for batch_idx, batch in enumerate(iterator):
                loss = self.calculate_loss(batch)

                tot_loss += loss.item()
                tot_batch += 1
                
                if self.show_progress_bar:
                    iterator.set_description('test loss {:.3f} '.format(tot_loss / tot_batch))
        
        print('test loss=', tot_loss / tot_batch)

        return tot_loss / tot_batch

    def calculate_loss(self, batch:torch.Tensor):
        v_truth = batch[:, 4].reshape(-1, 1)
        input_data = batch[:, 0:4]

        v = self.model(input_data)

        if self.is_binary_label:
            loss = self.BCEloss(v, v_truth)
        else:
            loss = self.MSEloss(torch.sigmoid(v), v_truth)
        
        return loss
    
    def save(self, tag: str):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optim.state_dict()}, os.path.join(self.train_dir, '{}.model'.format(tag)))