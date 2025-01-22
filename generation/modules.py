import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering

class Roberta_classifier(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.preLM = RobertaModel.from_pretrained('roberta-base')
        for p in self.preLM.parameters():
            p.requires_grad = True
        self.projection = nn.Linear(768, 128)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(128, class_num)

    def forward(self, t):
        text_features = self.preLM(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state[:, 0, :]
        projected = self.projection(text_features)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Llama_baseline(nn.Module):
    def __init__(self, config, class_num):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, 128)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(128, class_num)

    def forward(self, t):
        projected = self.projection(t)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Llama_baseline_generation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, 768)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(768, config.vocab_size)

    def forward(self, t):
        projected = self.projection(t)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def generate(self, ids, preLM, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5):
        for i in range(length):
            features = preLM(ids).last_hidden_state.float()
            projected = self.projection(features)
            x = self.gelu(projected)
            x = self.dropout(x)
            logits = self.fc(x)
            score = logits[:, -1, ids[0]]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[:, -1, ids[0]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_token), dim=-1)
            if list(next_token[0]) == [128001]:
                break
        return ids

class CBL(nn.Module):
    def __init__(self, config, concept_dim, tokenizer):
        super().__init__()
        self.cbl = nn.Linear(config.hidden_size, concept_dim)
        self.unsup = nn.Linear(config.hidden_size, 768)
        self.fc = nn.Linear(concept_dim + 768, config.vocab_size)
        self.relu = nn.ReLU()
        self.concept_dim = concept_dim
        self.tokenizer = tokenizer

    def forward(self, features):
        concepts = self.cbl(features)
        unsup_features = self.unsup(features)
        e = torch.cat((self.relu(concepts), unsup_features), dim=-1)
        return self.relu(concepts), unsup_features, self.fc(e)

    def generate(self, ids, preLM, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5):
        for i in range(length):
            features = preLM(ids).last_hidden_state.float()
            concepts = self.cbl(features)
            unsup_features = self.unsup(features)
            if intervene:
                for j in range(self.concept_dim):
                    concepts[0, :, j] = intervene[j]
            logits = self.fc(torch.cat((self.relu(concepts), unsup_features), dim=-1))
            score = logits[:, -1, ids[0]]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[:, -1, ids[0]] = score
            next_token_logits = logits[:, -1, :] / temp
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_token), dim=-1)
            if list(next_token[0]) == [128001]:
                break
        return ids, self.relu(concepts)[0]