from csv import DictWriter
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from simulator.interfaces.config import AreaAttributes, BuildingAttributes, FacilityAttributes, SimulatorConfig

def convert_bfs_xml_to_json(src_xml_path: Path, dst_dir_path: Path):
    dst_dir_path.mkdir(exist_ok=True)
    
    root = ET.parse(src_xml_path).getroot()

    # Extract areas as dicts
    area_dicts: list[dict[str, Any]] = []

    area_elems = filter(lambda elem: elem.tag == 'area', root)
    for area_elem in sorted(area_elems, key=lambda elem: elem.attrib['id']):
        
        facility_elems = filter(lambda child: child.tag == 'facility', area_elem)

        facilities = []

        for facility_elem in sorted(facility_elems, key=lambda elem: elem.attrib['id']):
            assert int(facility_elem.attrib['id']) == len(facilities), \
                "Area IDs must start from 0 and must be consecutive."

            params = \
                dict(
                    (param.attrib['name'].replace('-', '_'), param.attrib['value']) 
                    for param in filter(lambda child: child.tag == 'parameter', facility_elem)
            )
            facilities.append(FacilityAttributes(type=facility_elem.attrib["type"], parameters=params))

        area_dict = dict(
            name=area_elem.attrib['name'],
            facilities=facilities
        )
        if area_elem.attrib.get('capacity'):
            area_dict.update(capacity=float(area_elem.attrib.get('capacity')))
        if area_elem.attrib.get('temperature'):
            area_dict.update(initial_temperature=float(area_elem.attrib.get('temperature')))
            
        area_dicts.append(area_dict)
    
    # Write area env csvs
    area_env_elems = filter(lambda elem: elem.tag == 'area-environment', root)
    for area_env_elem in area_env_elems:
        area_id = int(area_env_elem.attrib['area-id'])
        area_name = area_dicts[area_id]['name'].replace(' ', '-')
        area_env_csv_path = dst_dir_path / f"{area_name}_area_env.csv"

        with area_env_csv_path.open('w') as f:
            writer = DictWriter(f, fieldnames=['people', 'heat_source'])
            writer.writeheader()
            for child in area_env_elem:
                writer.writerow(dict(
                    people=int(child.attrib['people']),
                    heat_source=float(child.attrib['heat-source'])
                ))

        area_dicts[area_id].update(area_environment_csv_path = area_env_csv_path)

    # Write external env csv
    ext_env_csv_path = dst_dir_path / f"ext_env.csv"
    with ext_env_csv_path.open('w') as f:
        writer = DictWriter(f, fieldnames=['solar_radiation', 'temperature', 'electric_price_unit'])
        writer.writeheader()

        ext_env_elem = next(filter(lambda elem: elem.tag == 'environment', root))

        for child in ext_env_elem:
            writer.writerow(dict(
                solar_radiation=float(child.attrib['solar-radiation']),
                temperature=float(child.attrib['temperature']),
                electric_price_unit=float(child.attrib['electric-price-unit'])
            ))
 
    # Write main json file
    start_time = datetime.strptime(ext_env_elem[0].attrib["time"], "%Y-%m-%d %H:%M")
    areas = [
        AreaAttributes(**area_dict) for area_dict in area_dicts
    ]
    output_json_path = dst_dir_path / f"simulator_config.json"
    output_json_path.write_text(
            SimulatorConfig(
                start_time=start_time,
                building_attributes=BuildingAttributes(areas=areas),
                external_environment_csv_path=ext_env_csv_path
            ).json(
                exclude_unset=True,
                exclude={
                    'external_enviroment_time_series': True, 
                    'building_attributes': {
                        'areas': {
                            '__all__': {
                                'area_environment_time_series': True
                            }
                        }
                    }
                },
                by_alias=True,
                indent=4, 
                separators=(',', ': ')
            ),
    )

    print(f"Converted {src_xml_path} to {dst_dir_path}!")
    

if __name__ == "__main__":
    xml_dir_path = Path("./data/xml/")
    json_dir_path = Path("./data/json/")

    # Convert example file
    convert_bfs_xml_to_json(xml_dir_path / "example/BFS_environment.xml", json_dir_path / "example")

    # Convert experimental files
    for xml_path in sorted(xml_dir_path.glob("*.xml")):
        convert_bfs_xml_to_json(xml_path, json_dir_path / xml_path.stem)
