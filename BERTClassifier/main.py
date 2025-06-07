from datetime import datetime
from typing import List
from fastapi import FastAPI, Depends, HTTPException
import torch
from pydantic import BaseModel
from torch import nn
import torch.nn.functional as F
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from database.models import Base
from database.db import SessionLocal, engine
from sqlalchemy.orm import Session
from analysis.correlation import compute_pearson_from_db

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# BERTClassifier 정의
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
        out = self.dropout(pooler) if self.dr_rate else pooler
        return self.classifier(out)

# BERTClassifier 객체만 만들고, 가중치 전체 로드
model = BERTClassifier(BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False), dr_rate=0.5).to(device)
model.load_state_dict(torch.load('model/kobert_finetuned.pt', map_location=device))
model.eval()

# 예측 함수
def predict(sentence):
    inputs = tokenizer.encode_plus(
        sentence,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    token_ids = inputs['input_ids'].to(device)           # [1, 64]
    segment_ids = inputs['token_type_ids'].to(device)     # [1, 64]
    valid_length = torch.tensor([inputs['attention_mask'].sum().item()], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(token_ids, valid_length, segment_ids)
        probs = F.softmax(output, dim=1)
        # print(f" 확률 분포: {probs.cpu().numpy()}")
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()
        positive_conf = probs[0][1].item()

    return predicted, confidence, positive_conf

class BertRequestDTO(BaseModel):
    date: datetime
    content: str

class NewsList(BaseModel):
    news: List[BertRequestDTO]

@app.post("/predict")
def predict_sentences(data: NewsList):
    results = []
    positive_confidences = []
    for item in data.news:
        content = item.content
        label, conf, positive_conf = predict(content)
        results.append({
            "date": item.date,
            "content": content,
            "label": label,
            "positive_conf": round(positive_conf, 4)
        })
        positive_confidences.append(positive_conf)

    avg_positive = sum(positive_confidences) / len(positive_confidences)
    final_label = "긍정" if avg_positive >= 0.5 else "부정"
    print(avg_positive, final_label)

    return {
        "result": final_label,
        "posProb": round(avg_positive, 4),
        "individual_results": results
    }

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

origins = [
    os.getenv("FASTAPI_ORIGIN"),
]

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # 허용할 origin 리스트
    allow_credentials=True,
    allow_methods=["*"],            # 모든 HTTP 메서드 허용
    allow_headers=["*"],            # 모든 헤더 허용
)


@app.get("/analyze")
def analyze(db: Session = Depends(get_db)):
    try:
        return compute_pearson_from_db(db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
