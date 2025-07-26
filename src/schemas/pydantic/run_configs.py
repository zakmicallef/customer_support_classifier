from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, computed_field

class DataConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rag_save_file_name: Optional[Path] = None
    test_factor: float

class LiveConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    priority_model_name: str
    queue_model_name: str
    confidence_on_low: float
    confidence_on_medium: float
    confidence_on_high: float

    priority_stop: bool

class TestConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    priority_model_name: str
    queue_model_name: str
    confidence_on_low: float
    confidence_on_medium: float
    confidence_on_high: float
    test_name: str

    @computed_field(return_type=dict)
    @property
    def confidence_map(self):
        return {
            'low': self.confidence_on_low,
            'medium': self.confidence_on_medium,
            'high': self.confidence_on_high,
        }