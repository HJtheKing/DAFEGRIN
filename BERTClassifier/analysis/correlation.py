from sqlalchemy.orm import Session
from database.models import DailyIndex
from scipy.stats import pearsonr

def compute_pearson_from_db(db: Session):
    data = db.query(DailyIndex).filter(
        DailyIndex.sentiment.isnot(None),
        DailyIndex.kospi.isnot(None)
    ).all()

    if len(data) < 2:
        raise ValueError("분석 가능한 데이터가 부족합니다.")

    sentiments = [d.sentiment for d in data]
    kospi_values = [d.kospi for d in data]

    correlation, p_value = pearsonr(sentiments, kospi_values)

    return {
        "correlation": round(correlation, 4),
        "p_value": round(p_value, 4),
        "count": len(data)
    }