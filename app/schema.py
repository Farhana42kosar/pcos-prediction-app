from typing import Optional
from pydantic import BaseModel

class PCOSInput(BaseModel):
    age_yrs: int
    weight_kg: float
    heightcm: float
    cycleri: int
    cycle_lengthdays: int
    hbgdl: float
    blood_group: Optional[str] = None   # 👈 change here
    marraige_status_yrs: float
