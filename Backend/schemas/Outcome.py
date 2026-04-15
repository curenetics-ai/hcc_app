from pydantic import BaseModel

class OutcomeCreate(BaseModel):
    patient_id: str
    outcome: int  # 1 = success, 0 = failure