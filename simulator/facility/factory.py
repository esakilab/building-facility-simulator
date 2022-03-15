from typing import Any, Type, TypeVar

from simulator.facility.facility_base import Facility


class FacilityFactory:
    T = TypeVar('T', bound=Facility)
    type_str_to_class: dict[str, Type[T]] = {}


    @staticmethod
    def create(type_str: str, parameters: dict[str, Any]) -> Facility:
        return FacilityFactory.type_str_to_class[type_str].parse_obj(parameters)


    @staticmethod
    def validate_parameters(type_str: str, parameters: dict[str, Any]):
        FacilityFactory.type_str_to_class[type_str].validate(parameters)


    @staticmethod
    def register(type_str: str):
        def _register(cls):
            assert issubclass(cls, Facility)

            FacilityFactory.type_str_to_class[type_str] = cls

            return cls
        
        return _register
