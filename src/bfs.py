from typing import Dict, List
from src.environment import AreaEnvironment, ExternalEnvironment
from src.area import Area
import xml.etree.ElementTree as ET


class BuildingFacilitySimulator:
    areas: Dict[str, Area] = {}
    ext_envs: List[ExternalEnvironment] = []
    area_envs: Dict[str, List[AreaEnvironment]] = {}


    def __init__(self, cfg_path: str):
        root = ET.parse(cfg_path).getroot()
        
        assert root.tag == "BFS", "invalid BFS XML"

        for child in root:
            if child.tag == 'area':
                self.areas[child.attrib['id']] = Area.from_xml_element(child)

            elif child.tag == 'environment':
                self.ext_envs = [
                    ExternalEnvironment.from_xml_element(elem) for elem in child
                ]

            elif child.tag == 'area-environment':
                area_id = child.attrib['area-id']
                self.area_envs[area_id] = [
                    AreaEnvironment.from_xml_element(elem) for elem in child
                ]

            else:
                print(f"Ignoring an element with unknown tag: {child}")

        
    def next_step(self) -> float:
        for t, ext_env in enumerate(self.ext_envs):
            total_power_consumption = 0.
            for area_id, area in self.areas.items():
                area.update(ext_env, self.area_envs[area_id][t])
                total_power_consumption += area.power_consumption

            yield total_power_consumption
            
