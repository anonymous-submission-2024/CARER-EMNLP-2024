import torch
from torch import nn

from models.text_model.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.text_model.utils import DotProductAttention, SingleHeadAttentionLayer, CustomAttentionLayer, ScaledDotProductAttention
from models.text_model.text_transformer import NMT_tran




from transformers import AutoModel, AutoTokenizer, AutoConfig

MODEL_NAME = 'yikuan8/Clinical-Longformer'


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class TextClassifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output



class TextModel(nn.Module):

    # Constructor class
    def __init__(self, model_name = MODEL_NAME, device = 'cuda'):
        super(TextModel, self).__init__()
        configuration = AutoConfig.from_pretrained(model_name)
        # configuration.hidden_dropout_prob = 0.5
        # configuration.attention_probs_dropout_prob = 0.3


        configuration.hidden_dropout_prob = 0.2
        configuration.attention_probs_dropout_prob = 0.2

        self.bert = AutoModel.from_pretrained(model_name,config = configuration).to(device)

        freezed_layers = [f'encoder.layer.{i}.' for i in range(6)]
        # print(freezed_layers)
        # print(freezed_layers)
        for name, param in self.bert.named_parameters():
            for layer_start in freezed_layers:
                # print(layer_start)
                if layer_start in str(name): # choose whatever you like here
                    # print(name, type(name))
                    param.requires_grad = False    
                    break        

        self.drop =nn.Dropout(0.2)
        # self.drop = nn.DR
        self.activation = nn.LeakyReLU()
        # self.bert.params = self.bert.to_fp16(self.bert.params)
    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        # print( self.bert(
        #   input_ids=input_ids,
        #   attention_mask=attention_mask
        # ).keys())
        # print( self.bert(
        #   input_ids=input_ids,
        #   attention_mask=attention_mask
        # ))
        # input_ids = input_ids.to(torch.float16)
        # attention_mask = attention_mask.to(torch.float16)
        hidden_state = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )['last_hidden_state']
        # print('hidden_state: ', hidden_state)

        #  Add a dropout layer
        hidden = self.drop(self.activation((hidden_state)))

        # # output = self.drop(pooled_output)
        # output = self.out(hidden)
        return hidden

# class FusionForgetGate(nn.Module):
#     def __init__(self, key_size, value_size):
#         super().__init__()

#         self.linear_forget = nn.Linear()



class FusionForgetGate(nn.Module):
    def __init__(self,key_size, value_size):
        super().__init__()

        self.linear_forget = nn.Linear(key_size + value_size,value_size)
        self.linear_memory  = nn.Linear(value_size, value_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, key,query):
        key_query_concat = torch.cat([query,key], dim  =-1)
        forget_projection  = self.sigmoid(self.linear_forget(key_query_concat))

        memory  =self.linear_memory(query)
    
        # memory = query
        filtered_vector = forget_projection * memory

        return filtered_vector


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation,
                 transformer_hidden_size = 768,
                 transformer_att_head_num = 4,text_embedding_size  = 128,vocab_size = 5000,
                 encoder_layers=4,transformer_dropout_rate=0.2):
        super().__init__()
        # print('hidden_size',hidden_size)
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 32)
        self.icd_seq_attention = DotProductAttention(256 + 150,32)

        # self.text_transformer = NMT_tran(transformer_hidden_size,
        #          transformer_att_head_num ,text_embedding_size ,vocab_size ,
        #          encoder_layers,transformer_dropout_rate)
        self.text_transformer = TextModel()
        transformer_hidden_size = 768
        self.icd_text_attention = CustomAttentionLayer(query_size=hidden_size,key_size=transformer_hidden_size,value_size=1,attention_size=256)
        # self.icd_cls_attention = CustomAttentionLayer(query_size=hidden_size,key_size=1,value_size=1,attention_size=256)

        self.icd_text_attention = ScaledDotProductAttention(query_dim=hidden_size,key_dim=transformer_hidden_size,value_dim=1)

        # self.text_fc = nn.Linear(transformer_hidden_size * 2, 256)
        # self.text_fc = nn.Linear(transformer_hidden_size * 2 , 256)
        # self.text_fc = nn.Linear(transformer_hidden_size  * 2 , 256)
        self.text_fc = nn.Linear(128  * 2 , 128)

        # self.text_fc_2 = nn.Linear(transformer_hidden_size , 256)

        # self.classifier = Classifier(hidden_size + 256 , output_size, dropout_rate, activation)
        self.classifier = Classifier(128 * 2 , output_size, dropout_rate, activation)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

        # self.text_classifier = Classifier(768,output_size, dropout_rate, activation)
        # self.icd_classifier = Classifier(hidden_size,output_size, dropout_rate, activation)




        self.contrastive_raw = nn.Linear(hidden_size , 128)

        self.contrastive_text = nn.Linear(transformer_hidden_size , 128)

    def forward(self, code_x, divided, neighbors, lens, note, note_mask, contrastive_mode = False):
        note = note[:,:512]
        note_mask = note_mask[:,:512]
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        output_icds = []
        i = 0
        max_len = 0
        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
            i +=1
            no_embeddings_i_prev = None
            output_i = []
            h_t = None

            historical_embeddings = []
            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                historical_embeddings.append(co_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            if len(output_i) > max_len:
                max_len = len(output_i)
                # print(max_len)

            output_icd = torch.vstack(output_i)
            # print('output_icd',output_icd.shape)
            
            icd_self_att = self.attention(output_icd)

            historical_embeddings = torch.stack(historical_embeddings)
            icd_embeddings = historical_embeddings.max(dim = 0).values

            output.append(icd_self_att)
            output_icds.append(output_icd)

        padded_output_icds = []

        for output_icd in output_icds:
            padded_output = torch.nn.functional.pad(input=output_icd, pad=(0,0, 0,max_len - output_icd.shape[0]), mode='constant', value=0)
            padded_output_icds.append(padded_output)
        icd_seq_output = torch.stack(padded_output_icds)
        patient_output = torch.vstack(output)
        text_embedding = self.text_transformer(note,note_mask)
        token_embeddings = text_embedding[:,1:,:]

        cls_embeddings = text_embedding[:,0,:]
        patient_text_att = self.icd_text_attention(queries=patient_output,keys=token_embeddings,values=token_embeddings,mask=note_mask[:,1:])

        # cls_embeddings_masked = cls_embeddings
        cls_embeddings = self.contrastive_text(cls_embeddings)
        patient_text_att = self.contrastive_text(patient_text_att)
        patient_output = self.contrastive_raw(patient_output)


        text_rep = torch.cat([patient_text_att, cls_embeddings],dim = -1)
        text_rep = self.text_fc(text_rep)
        text_rep = self.relu(text_rep)

        output = torch.cat([patient_output,text_rep],dim=-1) 


        # output = patient_output
        # output = output.squeeze(1)
        output = self.classifier(output)

        if contrastive_mode:
            return output, (patient_output, cls_embeddings)

        # icd_output = patient_output.squeeze(1)
        # icd_output = self.icd_classifier(icd_output)

        # text_output = self.text_classifier(cls_embeddings)


        return output

        # return final_output

    def forward_with_text_rep(self, code_x, divided, neighbors, lens, note, note_mask):
        # print(note.shape, note_mask.shape)
        note = note[:,:4096]
        note_mask = note_mask[:,:4096]
        # print(note.shape, note_mask.shape)
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        i = 0
        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
            i +=1
            no_embeddings_i_prev = None
            output_i = []
            h_t = None

            historical_embeddings = []
            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                historical_embeddings.append(co_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                output_i.append(output_it)
            output_i = self.attention(torch.vstack(output_i))

            historical_embeddings = torch.stack(historical_embeddings)
            icd_embeddings = historical_embeddings.max(dim = 0).values

            output.append(output_i)
        patient_output = torch.vstack(output)

        # text_embedding = self.text_transformer(note,note_mask)
        # token_embeddings = text_embedding[:,1:,:]

        # cls_embeddings = text_embedding[:,0,:]
        # icd_text_att, id = token_embeddings.max(dim = 1)
        # print(id)
        # icd_text_att = self.icd_text_attention(queries=patient_output,keys=token_embeddings,values=token_embeddings,mask=note_mask[:,1:])
        # text_rep = torch.cat([icd_text_att, cls_embeddings],dim = -1)
        # text_rep = self.text_fc(text_rep)
        # text_rep = self.dropout(self.relu(text_rep))
        # text_rep = icd_text_att
        output = torch.cat([patient_output,text_rep],dim=-1) 

        output = self.classifier(output)


        icd_output = patient_output.squeeze(1)
        icd_output = self.icd_classifier(icd_output)

        text_output = self.text_classifier(cls_embeddings)   
        # print(output.shape, icd_output.shape, text_output.shape)    
        return output, (icd_output, text_output)


    # def forward_with_interpolation(self, code_x, divided, neighbors, lens, note, note_mask):
    #     # print(note.shape, note_mask.shape)
    #     note = note[:,:4096]
    #     # note =  self.embedding_layer()
    #     note_mask = note_mask[:,:4096]
    #     # print(note.shape, note_mask.shape)
    #     embeddings = self.embedding_layer()
    #     c_embeddings, n_embeddings, u_embeddings = embeddings
    #     output = []
    #     i = 0
    #     for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
    #         i +=1
    #         no_embeddings_i_prev = None
    #         output_i = []
    #         h_t = None

    #         historical_embeddings = []
    #         for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
    #             co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
    #             historical_embeddings.append(co_embeddings)
    #             output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
    #             no_embeddings_i_prev = no_embeddings
    #             output_i.append(output_it)
    #         output_i = self.attention(torch.vstack(output_i))

    #         historical_embeddings = torch.stack(historical_embeddings)
    #         icd_embeddings = historical_embeddings.max(dim = 0).values

    #         output.append(output_i)
    #     patient_output = torch.vstack(output)

    #     text_embedding = self.text_transformer(note,note_mask)
    #     token_embeddings = text_embedding[:,1:,:]

    #     cls_embeddings = text_embedding[:,0,:]
    #     # icd_text_att, id = token_embeddings.max(dim = 1)
    #     # print(id)
    #     icd_text_att = self.icd_text_attention(queries=patient_output,keys=token_embeddings,values=token_embeddings,mask=note_mask[:,1:])
    #     # text_rep = icd_text_att
    #     text_rep = torch.cat([icd_text_att, cls_embeddings],dim = -1)
    #     text_rep = self.text_fc(text_rep)
    #     text_rep = self.dropout(self.relu(text_rep))
    #     # text_rep = icd_text_att
    #     output = torch.cat([patient_output,text_rep],dim=-1) 
    #     # print('concat output: ', output)
    #     # output = patient_output
    #     # print(all_text_hiddens.shape)
    #     # print(first_hidden.shape)
    #     # output = output.squeeze(1)
    #     # print(output.shape)
    #     output = self.classifier(output)
    #     output = self.classifier(output)

    #     icd_output = patient_output.squeeze(1)
    #     icd_output = self.icd_classifier(icd_output)

    #     concat_output = torch.concat([output,icd_output], dim  =-1)

    #     final_output = self.final_interpolation_layer(concat_output)

    #     # text_output = self.text_classifier(cls_embeddings)   
    #     # print(output.shape, icd_output.shape, text_output.shape)    
    #     return final_output, (output, icd_output)
