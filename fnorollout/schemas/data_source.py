from pydantic import BaseModel

from fnorollout.constants import DataSourceType


class DataSource(BaseModel):
    """A single named data source, as defined under configs/data_sources/*.yaml."""

    name: str
    type: DataSourceType
    channels: list[str]
