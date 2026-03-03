# api/schemas.py

from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):

    Age: int
    Gender: str
    City: str
    Employment_Type: str

    Annual_Income: Optional[float] = None
    Credit_Score: Optional[float] = None
    Tenure_Months: int
    Avg_Monthly_Balance: Optional[float] = None
    Balance_Change_Ratio: Optional[float] = None

    Mobile_App_Logins: Optional[float] = None
    UPI_Transactions: Optional[float] = None
    ATM_Withdrawals: Optional[float] = None
    NetBanking_Usage: Optional[float] = None

    Complaint_Tickets: Optional[float] = None
    Call_Center_Interactions: Optional[float] = None

    Has_Credit_Card: int
    Has_Loan: int
    Has_Insurance: int