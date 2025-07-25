from pathlib import Path
import yaml
from schemas.pydantic.run_configs import DataConfig, LiveConfig, TestConfig

def load_live_config() -> LiveConfig:
    with open(Path("./config/live_config.yaml"), "r") as f:
        raw_cfg = yaml.safe_load(f)
    return LiveConfig(**raw_cfg)

def load_test_config() -> TestConfig:
    with open(Path("./config/test_config.yaml"), "r") as f:
        raw_cfg = yaml.safe_load(f)
    return TestConfig(**raw_cfg)

def load_data_config() -> DataConfig:
    with open(Path("./config/data_config.yaml"), "r") as f:
        raw_cfg = yaml.safe_load(f)
    return DataConfig(**raw_cfg)
