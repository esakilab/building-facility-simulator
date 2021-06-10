from xml.etree.ElementTree import Element
from typing import Type
from src.facility.facility_base import Facility
from src.facility.hvac import HVAC
from src.facility.pv_station import PVStation
from src.facility.electric_storage import ElectricStorage


def xml_element_to_facility(elem: Element) -> Facility:
    type_to_cls: dict[str, Type[Facility]] = {
        cls.TYPE_STR: cls for cls in [PVStation, HVAC, ElectricStorage]
    }

    return type_to_cls[elem.attrib['type']].from_xml_element(elem)