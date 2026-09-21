from pydantic import BaseModel, ConfigDict

from fnorollout.schemas.data_source import LocalDataSource


class Rollout2DConfig(BaseModel):
    """
    Schema for the fully composed Hydra config for rollouts (configs/rollout_config.yaml).
    """

    model_config = ConfigDict(extra="allow")

    device: str
    rollout_steps: int
    start_index: int = 0
    artifacts: dict[str, str]
    data_sources: LocalDataSource
    output_path: str
