from pydantic import BaseModel, ConfigDict, model_validator

from fnorollout.schemas.data_source import DataSource


class DataConfig(BaseModel):
    """Schema for the `data` config group (configs/data/*.yaml)."""

    model_config = ConfigDict(extra="allow")

    sources: dict[str, DataSource]
    test_sources: dict[str, DataSource] | None = None


class ModelConfig(BaseModel):
    """Schema for the `model` config group (configs/model/*.yaml)."""

    model_config = ConfigDict(extra="allow")


class TrainingConfig(BaseModel):
    """Schema for the `training` config group (configs/training/*.yaml)."""

    model_config = ConfigDict(extra="allow")


class Config(BaseModel):
    """Schema for the fully composed Hydra config (configs/config.yaml)."""

    model_config = ConfigDict(extra="allow")

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: dict[str, object]
    eval: dict[str, object]

    seed: int
    device: str

    @model_validator(mode="after")
    def check_physics_constraints(self):
        pass
