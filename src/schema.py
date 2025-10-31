from pydantic import BaseModel, conlist

# Expect 4 iris features; batch of rows allowed
class PredictRequest(BaseModel):
    rows: list[conlist(float, min_items=4, max_items=4)]
