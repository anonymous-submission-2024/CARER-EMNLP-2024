import torch
from torch import nn

# from models.text_model.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention, SingleHeadAttentionLayer, CustomAttentionLayer, ScaledDotProductAttention
from models.text_transformer import NMT_tran

from models.hitanet import  HitaNet
from models.t_lstm import TimeLSTM


from transformers import AutoModel, AutoTokenizer, AutoConfig

MODEL_NAME = 'yikuan8/Clinical-Longformer'

class DemographicMLP(nn.Module):

    def __init__(self, input_size, output_size, dropout_rate=0.):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, output_size // 2)
        self.linear_2 = nn.Linear(output_size//2, output_size)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.linear_1(x)
        # if self.activation is not None:
        output = self.activation(output)
        output = self.dropout(x)
        output = self.linear_2(output)

        return output

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



# class FusionForgetGate(nn.Module):
#     def __init__(self,key_size, value_size):
#         super().__init__()

#         self.linear_forget = nn.Linear(key_size + value_size,value_size)
#         self.linear_memory  = nn.Linear(value_size, value_size)

#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()

#     def forward(self, key,query):
#         key_query_concat = torch.cat([query,key], dim  =-1)
#         forget_projection  = self.sigmoid(self.linear_forget(key_query_concat))

#         memory  =self.linear_memory(query)
    
#         # memory = query
#         filtered_vector = forget_projection * memory

#         return filtered_vector


class Model(nn.Module):
    def __init__(self,  code_size, lab_values_input_size, demographic_input_size, hidden_size, dropout,  num_hitanet_layers, num_heads,  transformer_hidden_size = 768, activation = None):
        super().__init__()
        # print('hidden_size',hidden_size)
        # self.icd_seq_attention = DotProductAttention(256 + 150,32)
        self.hitanet =  HitaNet(code_size, hidden_size, dropout, dropout, num_layers, num_heads, max_pos =128)
        self.t_lstm = TimeLSTM(lab_values_input_size, hidden_size)
        self.d_mlp = DemographicMLP(demographic_input_size, hidden_size)
        self.text_transformer = TextModel()
        transformer_hidden_size = 768

        self.icd_text_attention = ScaledDotProductAttention(query_dim=hidden_size,key_dim=transformer_hidden_size,value_dim=1)

        self.text_fc = nn.Linear(hidden_size + transformer_hidden_size , hidden_size)
        self.reasoning_fusion_fc = nn.Linear(hidden_size + transformer_hidden_size , hidden_size)


        # self.classifier = Classifier(hidden_size + 256 , output_size, dropout_rate, activation)
        self.classifier = Classifier(128 + transformer_hidden_size , output_size, dropout, activation)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()

        # self.text_classifier = Classifier(768,output_size, dropout_rate, activation)
        # self.icd_classifier = Classifier(hidden_size,output_size, dropout_rate, activation)




        # self.contrastive_raw = nn.Linear(hidden_size , 128)

        # self.contrastive_text = nn.Linear(transformer_hidden_size , 128)

    def forward(self, code_x,code_x_mask, lab_x, num_visits, demo_x, note, note_mask, reasoning, reasoning_mask, seq_time_step, alignment_mode = False):
        note = note[:,:2048]
        note_mask = note_mask[:,:2048]

        reasoning = reasoning[:,:2048]
        reasoning_mask = reasoning_mask[:,:2048]

        # patient_output = torch.vstack(output)
        icd_output = self.hitanet(code_x, code_x_mask,num_visits,seq_time_step)
        lab_output = self.t_lstm(lab_x, seq_time_step,num_visits)
        demo_output = self.d_mlp(demo_x)

        patient_output = (icd_output + lab_output + demo_output) / 3
        text_embedding = self.text_transformer(note,note_mask)
        token_embeddings = text_embedding[:,1:,:]

        patient_text_att = self.icd_text_attention(queries=patient_output,keys=token_embeddings,values=token_embeddings,mask=note_mask[:,1:])


        patient_output = torch.cat([patient_output,patient_text_att],dim=-1) 
        patient_output = self.relu(self.text_fc(patient_output))


        reasoning_embedding = self.text_transformer(reasoning,reasoning_mask)
        reasoning_cls_embeddings = reasoning_embedding[:,0,:]

        patient_reasoning_att = self.icd_text_attention(queries=patient_output,keys=reasoning_embeddings,values=reasoning_embeddings,mask=reasoning_mask[:,1:])
        patient_output_fused = torch.cat([patient_output,patient_reasoning_att],dim=-1) 
        patient_output = self.relu(self.reasoning_fusion_fc(patient_output_fused))


        output = self.classifier(patient_output)

        if alignment_mode:
            return output, (patient_output, reasoning_cls_embeddings)

        # icd_output = patient_output.squeeze(1)
        # icd_output = self.icd_classifier(icd_output)

        # text_output = self.text_classifier(cls_embeddings)


        return output

        # return final_output

