import torch
import random
import numpy as np
# import torch.nn.functional as F

class MentorNet_nn(torch.nn.Module):
    def __init__(self,  label_embedding_size=2, 
                        epoch_embedding_size=5, 
                        num_fc_nodes=20,
                        device="cpu"):
        """
        Args:
            label_embedding_size: the embedding size for the label feature.
            
            epoch_embedding_size: the embedding size for the epoch feature.
            
            num_fc_nodes: number of hidden nodes in the fc layer.
        Input:
            input_features: a [batch_size, 4] tensor. Each dimension corresponds to
            0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
            where epoch is an integer between 0 and 99 (the first and the last epoch).

            input_feature: B x 4
        Output:
            v: [batch_size, 1] weight vector.s
        """
        super(MentorNet_nn, self).__init__()
        
        self.device = device

        self.label_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=label_embedding_size).to(device)
        
        self.epoch_embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=epoch_embedding_size).to(device)

        self.bi_lstm_cell = torch.nn.LSTM(input_size=2, hidden_size=1,bidirectional=True,batch_first=True,num_layers=1).to(device)

        self.feat_size = label_embedding_size + epoch_embedding_size + 2

        self.fc1 = torch.nn.Linear(self.feat_size, num_fc_nodes).to(device)
        self.fc2 = torch.nn.Linear(num_fc_nodes, 1, bias=True).to(device)
        
    def forward(self, input_features):
        input_features = input_features.to(self.device)
        losses = input_features[:, 0]
        loss_diffs = input_features[:, 1]

        lstm_inputs = torch.stack([losses, loss_diffs], dim=-1).to(self.device).to(torch.float32)

        if len(losses.shape) <= 1:
            num_steps = 1
            lstm_inputs.unsqueeze_(1)
        else:
            num_steps = int(losses.size()[1])

        # lstm_inputs should be B x N x 2 
        # where N is the num_steps, B is the batch size


        lstm_output, _ = self.bi_lstm_cell(lstm_inputs)

        # lstm_output should be B x N x 2 
        # where '2' is due to bidirectional setting

        loss_variance = lstm_output.sum(1) # B x 2

        labels = input_features[:, 2].reshape((-1, 1)).to(torch.int64)
        
        epochs = input_features[:, 3].reshape((-1, 1)).to(torch.int64)
        epochs = torch.min(epochs, torch.ones([epochs.size()[0], 1], dtype=torch.int64).to(self.device) * 99).to(self.device)

        # epoch_embedding.weight.requires_grad = False

        label_inputs = self.label_embedding(labels).squeeze(1) # B x D
        epoch_inputs = self.epoch_embedding(epochs).squeeze(1) # B x D

        # print(label_inputs.size(), epoch_inputs.size(), loss_variance.size())

        feat = torch.cat([label_inputs, epoch_inputs, loss_variance], -1).to(self.device)

        fc_1 = self.fc1(feat)
        output_1 = torch.tanh(fc_1)
        
        return self.fc2(output_1)


class MentorNet(torch.nn.Module):
    def __init__(self,  burn_in_epoch=18,
                        fixed_epoch_after_burn_in = True,
                        loss_moving_average_decay=0.9,
                        device="cpu"):
        """
            The MentorNet to train with the StudentNet.
        Args:
            burn_in_epoch: the number of burn_in_epoch. In the first burn_in_epoch, all samples have 1.0 weights.

            fixed_epoch_after_burn_in: whether to fix the epoch after the burn-in.
            
            loss_moving_average_decay: the decay factor to compute the moving average.
        Input:
            epoch: a tensor [batch_size, 1] representing the training percentage. Each epoch is an integer between 0 and 99.

            loss: a tensor [batch_size, 1] representing the sample loss.
    
            labels: a tensor [batch_size, 1] representing the label. Every label is set to 0 in the current version.

            loss_p_percentile: a 1-d tensor of size 100, where each element is the p-percentile at that epoch to compute the moving average.

            example_dropout_rates: a 1-d tensor of size 100, where each element is the dropout rate at that epoch. Dropping out means the probability of setting sample weights to zeros proposed in Liang, Junwei, et al. "Learning to Detect Concepts from Webly-Labeled Video Data." IJCAI. 2016.
        """
        super(MentorNet, self).__init__()
        
        self.device = device

        self.fixed_epoch_after_burn_in = fixed_epoch_after_burn_in

        self.burn_in_epoch = burn_in_epoch

        self.loss_moving_average_decay = loss_moving_average_decay

        self.mentor = MentorNet_nn(device=device)

        self.loss_moving_avg = None

    def forward(self, epoch, loss, labels, loss_p_percentile, example_dropout_rates):
        # epoch : B x 1
        # loss  : B x 1
        # labels: B x 1
        # loss_p_percentile: 100
        # example_dropout_rates: 100

        burn_in_epoch = torch.tensor([[self.burn_in_epoch]] * epoch.shape[0]).to(self.device)

        if not self.fixed_epoch_after_burn_in:
            cur_epoch = epoch
        else:
            cur_epoch = epoch.min(burn_in_epoch)
        
        # cur_epoch : B x 1

        v_ones = torch.ones(loss.size(), dtype=torch.float32).to(self.device)
        
        v_zeros = torch.zeros(loss.size(), dtype=torch.float32).to(self.device)

        upper_bound = torch.where(cur_epoch < burn_in_epoch - 1, v_ones, v_zeros).to(self.device)
        
        # TODO dangerous here
        this_dropout_rate = example_dropout_rates[cur_epoch][0][0]

        # TODO dangerous here
        this_percentile = loss_p_percentile[cur_epoch].squeeze()

        percentile_loss = torch.tensor(np.percentile(loss.cpu(), this_percentile.cpu()), dtype=torch.float32).unsqueeze(-1).to(self.device)

        # percentile_loss : B x 1

        if self.loss_moving_avg is None:
            self.loss_moving_avg = (1 - self.loss_moving_average_decay) * percentile_loss
        else:
            self.loss_moving_avg = self.loss_moving_avg * self.loss_moving_average_decay + (1 - self.loss_moving_average_decay) * percentile_loss

        # loss_moving_avg : B x 1

        # print(loss.size())

        input_data = torch.stack([loss, self.loss_moving_avg, labels, cur_epoch.to(torch.float32)], 1).squeeze(-1).to(self.device)

        # print(input_data.size())

        v = self.mentor(input_data).sigmoid().max(upper_bound)

        # print(torch.ceil(v.size()[0] * (1 - this_dropout_rate)))

        dropout_num = int(torch.ceil(v.size()[0] * (1 - this_dropout_rate)).item())

        idx = torch.tensor(random.sample(range(v.size()[0]), dropout_num), dtype=torch.int64).to(self.device)

        dropout_v = torch.zeros(v.size()[0]).to(self.device)
        dropout_v[idx] = 1

        # dropout_v.dot()

        return (v.squeeze() * (dropout_v)).unsqueeze(-1)
