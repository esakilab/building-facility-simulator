from __future__ import annotations
from csv import DictReader
from itertools import repeat
from typing import Iterator, Iterator, Type, TypeVar, Union
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field, FilePath, validator

from simulator.area import Area
from simulator.environment import AreaEnvironment, BuildingEnvironment, ExternalEnvironment
from simulator.facility.facility_base import Facility
from simulator.facility.factory import FacilityFactory


def check_at_least_one(target: dict[str, Any], field1: str, field2: str):
    if field1 not in target and field2 not in target:
        raise ValueError(f'Either one of {field1} and {field2} is required.')


M = TypeVar('M', bound=BaseModel)
class CsvModelIterator(Iterator[M]):
    def __init__(self, csv_path: Path, ModelType: Type[M]):
        self.file = csv_path.open()
        self.ModelType = ModelType
        self.header = self.file.readline().strip('\n').split(',')
    
    def __iter__(self) -> CsvModelIterator:
        return self
    
    def __next__(self) -> M:
        row = self.file.readline().split(',')
        return self.ModelType.parse_obj(
            {key: value for key, value in zip(self.header, row)}
        )


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
    area_environment_time_series: Optional[list[AreaEnvironment]] = None
    area_environment_csv_path: Optional[FilePath] = None


    def to_area(self) -> Area:
        return Area(
            name=self.name,
            facilities=list(map(FacilityAttributes.to_facility, self.facilities)),
            simulate_temperature=bool(
                self.capacity and (self.area_environment_time_series or self.area_environment_csv_path)),
            capacity=self.capacity or -1,
            temperature=self.initial_temperature
        )


class BuildingAttributes(BaseModel):
    areas: list[AreaAttributes]


class SimulatorConfig(BaseModel):
    start_time: datetime
    building_attributes: BuildingAttributes
    external_enviroment_time_series: Optional[list[ExternalEnvironment]] = None
    external_environment_csv_path: Optional[FilePath] = None

    
    @validator('external_enviroment_time_series')
    def check_external_environment(cls, v, values):
        check_at_least_one(values, 'external_environment_time_series', 'external_environment_csv_path')
        return v

    
    def get_env_iter(self) -> Iterator[BuildingEnvironment]:
        if self.external_enviroment_time_series:
            ext_env_iter = self.external_enviroment_time_series
        else:
            ext_env_iter = CsvModelIterator(self.external_environment_csv_path, ExternalEnvironment)
        
        def get_area_env_from_area_attr(area_attr: AreaAttributes) -> Iterator[tuple[Optional[AreaEnvironment], ...]]:
            if area_attr.area_environment_time_series:
                return area_attr.area_environment_time_series
            elif area_attr.area_environment_csv_path:
                return CsvModelIterator(area_attr.area_environment_csv_path, AreaEnvironment)
            return repeat(None)
        
        area_envs_iter = zip(*map(get_area_env_from_area_attr, self.building_attributes.areas))
        
        return map(
            lambda ext_areas: BuildingEnvironment(external=ext_areas[0], areas=list(ext_areas[1])), zip(ext_env_iter, area_envs_iter))
