import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KoBERT 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 데이터 로딩
dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("ratings_test.txt", field_indices=[1,2], num_discard_samples=1)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return len(self.labels)

# 하이퍼파라미터
max_len = 64
batch_size = 64
num_epochs = 3
learning_rate = 5e-5
warmup_ratio = 0.1
max_grad_norm = 1
log_interval = 200

# 데이터 전처리
tok = tokenizer.tokenize
data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)
train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=0)

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

# 모델, optimizer, scheduler 준비
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    acc = (max_indices == Y).sum().item() / max_indices.size(0)
    return acc

# Best 모델 저장을 위한 변수
best_test_acc = 0.0

if __name__ == "__main__":
    for e in range(num_epochs):
        train_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {:.4f} train acc {:.4f}".format(
                    e+1, batch_id+1, loss.item(), train_acc / (batch_id+1)))

        print("epoch {} train acc {:.4f}".format(e+1, train_acc / (batch_id+1)))

        # 평가
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)

        epoch_test_acc = test_acc / (batch_id + 1)
        print("epoch {} test acc {:.4f}".format(e+1, epoch_test_acc))

        # Best 모델만 저장
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save(model.state_dict(), 'kobert_finetuned.pt')
            print(f"✔️ Saved best model at epoch {e+1} with acc {best_test_acc:.4f}")
            # 추가 확인: classifier 포함 여부
            print("✔️ Saved keys:", [k for k in model.state_dict().keys() if 'classifier' in k])
