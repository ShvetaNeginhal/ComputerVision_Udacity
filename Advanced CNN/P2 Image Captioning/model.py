import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        #Embedding layer 
        self.embed = nn.Embedding(vocab_size,embed_size)
        
        # The LSTM Layer
        #params - input size, hidden units, no of LSTM layers,bias,input & output will have batch size as 1st dimension,
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)
        
        #Final output layer
        self.linear = nn.Linear(hidden_size,vocab_size)
        
        #initialize hidden layer with zeros
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
        
    def forward(self, features, captions):
        
        ## Discard the <end> word to avoid predicting when <end> is the input of the RNN
        #captions = captions[:, :-1]
        
        #Create embedding vector for each word in caption
        caption_embedding = self.embed(captions[:,:-1])
        
        #concatenate feature and caption --> input 
        embeddings = torch.cat((features.unsqueeze(1), caption_embedding), 1)
        
        lstm_out, self.hidden = self.lstm(embeddings)
       
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            res.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)

        return res