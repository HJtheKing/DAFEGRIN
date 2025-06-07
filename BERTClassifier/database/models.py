from sqlalchemy import Column, BigInteger, Date, Float, Boolean
from database.db import Base

class DailyIndex(Base):
    __tablename__ = "daily_index"

    daily_index_id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    daily_index_date = Column(Date, nullable=False)
    kospi = Column(Float, nullable=True)
    is_pos = Column(Boolean, nullable=True)
    sentiment = Column(Float, nullable=True)
