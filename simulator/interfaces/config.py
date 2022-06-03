from csv import DictReader
from typing import Union
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field, validator

from simulator.area import Area
from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.facility.facility_base import Facility
from simulator.facility.factory import FacilityFactory


def read_csv_as_list_of_dicts(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path) as f:
        return [row for row in DictReader(f)]


class FacilityAttributes(BaseModel):
    facility_type: str = Field(alias="type")
    parameters: dict[str, Union[int, float, str]]


    @validator('parameters')
    def check_parameters(cls, v, values):
        FacilityFactory.validate_parameters(values["facility_type"], v)
        return v


    def to_facility(self) -> Facility:
        return FacilityFactory.create(self.facility_type, self.parameters)


class AreaAttributes(BaseModel):
    name: str
    # If not specified, the temperature will not be simulated.
    capacity: Optional[float]
    initial_temperature: float = 25.
    facilities: list[FacilityAttributes]
    area_environment_time_series: list[AreaEnvironment] = []
    area_environment_csv_path: Optional[Path] = None

    
    def __init__(self, **data: Any):
        if (csv_path := data.get('area_environment_csv_path')):
            data.update(
                area_environment_time_series=read_csv_as_list_of_dicts(csv_path))
        super().__init__(**data)


    def to_area(self) -> Area:
        return Area(
            name=self.name,
            facilities=list(map(FacilityAttributes.to_facility, self.facilities)),
            simulate_temperature=bool(self.capacity) and bool(self.area_environment_time_series),
            capacity=self.capacity or -1,
            temperature=self.initial_temperature
        )


class BuildingAttributes(BaseModel):
    areas: list[AreaAttributes]


class SimulatorConfig(BaseModel):
    start_time: datetime
    building_attributes: BuildingAttributes
    external_enviroment_time_series: list[ExternalEnvironment] = []
    external_environment_csv_path: Optional[Path] = None

    
    def __init__(self, **data: Any):
        if (csv_path := data.get('external_environment_csv_path')):
            data.update(
                external_enviroment_time_series=read_csv_as_list_of_dicts(csv_path))
        super().__init__(**data)
