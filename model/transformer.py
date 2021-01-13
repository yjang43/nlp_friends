import math
import torch
import torch.nn as nn
import torch.functional as F

class DialogTransformer(nn.Module):
    """ Model to train conversational data on
    """

    def __init__(self, embedding, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        d_model = embedding.embedding_dim
        self.embedding = embedding      # pre-trained model: word2vec
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=0.1)      # hyper-parameter can change anytimes
        self.decoder_output = nn.Linear(embedding.embedding_dim, embedding.num_embeddings)

    def forward(self, src, tgt):
        """ Mostly following implementation of "Attention is all you need"
        src: (M x B x E) tensor
        tgt: (M' x B x E) tensor
             Dimension of training and inference stage can be different!
        """
        inputs = self.embedding(src)        # prepare inputs
        inputs = self.pos_encoder(inputs)

        outputs = self.embedding(tgt)       # prepare outputs
        outputs = self.pos_encoder(outputs)

        tgt_mask = self.generate_mask(outputs.size()[0])    # masking only for target

        output = self.transformer(inputs, outputs, tgt_mask=tgt_mask)   # transformer
        output = self.decoder_output(output)    # linear docoder output layer

        return output

    def generate_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
               

class PositionalEncoding(nn.Module):
    """ Encode positional information
    retrieved from tutorial (https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    *** note that d_model value needs to be even number
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# We can assume the dimension of input and output data the same
# M x B x E
# <f> filler token will be included

# embedding -> pos -> transformer -> pre_softmax_linear_transformation -> cross entropy

