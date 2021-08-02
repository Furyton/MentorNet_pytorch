import torch
import torch.nn.functional as F

class MentorNet_nn(torch.nn.Module):
    def __init__(self,  label_embedding_size=2, 
                        epoch_embedding_size=5, 
                        num_fc_nodes=20):
        """
        Args:
            label_embedding_size: the embedding size for the label feature.
            
            epoch_embedding_size: the embedding size for the epoch feature.
            
            num_fc_nodes: number of hidden nodes in the fc layer.
        Output:
            v: [batch_size, 1] weight vector.s
        """
        super(MentorNet_nn, self).__init__()
        
        self.label_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=label_embedding_size)
        
        self.epoch_embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=epoch_embedding_size)

        self.bi_lstm_cell = torch.nn.LSTM(input_size=2, hidden_size=1,bidirectional=True,batch_first=True,num_layers=1)

        self.feat_size = label_embedding_size + epoch_embedding_size + 2

        self.fc1 = torch.nn.Linear(self.feat_size, num_fc_nodes)
        self.fc2 = torch.nn.Linear(num_fc_nodes, 1, bias=True)
        
    def forward(self, input_features):
        """
        Input:
            input_features: a [batch_size, 4] tensor. Each dimension corresponds to
            0: loss, 1: loss difference to the moving average, 2: label and 3: epoch,
            where epoch is an integer between 0 and 99 (the first and the last epoch).

            input_feature: B x 4
        Returns:
            v: [batch_size, 1] weight vector.
        """

        losses = input_features[:, 0]
        loss_diffs = input_features[:, 1]

        lstm_inputs = torch.stack([losses, loss_diffs], dim=-1)

        if len(losses.shape) <= 1:
            num_steps = 1
            lstm_inputs.unsqueeze_(1)
        else:
            num_steps = int(losses.size()[1])

        # lstm_inputs should be B x N x 2 
        # where N is the num_steps, B is the batch size

        lstm_output, _ = self.bi_lstm_cell(lstm_inputs)

        # lstm_output should be B x N x 2 
        # where '2' is due to bidirectory setting

        loss_variance = lstm_output.sum(1) # B x 2

        labels = input_features[:, 2].reshape((-1, 1)).to(torch.int64)
        
        epochs = input_features[:, 3].reshape((-1, 1)).to(torch.int64)
        epochs = torch.min(epochs, torch.ones([epochs.size()[0], 1], dtype=torch.int64) * 99)

        # epoch_embedding.weight.requires_grad = False

        label_inputs = self.label_embedding(labels).squeeze(1) # B x D
        epoch_inputs = self.epoch_embedding(epochs).squeeze(1) # B x D

        # print(label_inputs.size(), epoch_inputs.size(), loss_variance.size())

        feat = torch.cat([label_inputs, epoch_inputs, loss_variance], -1)

        fc_1 = self.fc1(feat)
        output_1 = torch.tanh(fc_1)
        
        return self.fc2(output_1)