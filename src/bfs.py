from src.environment import AreaEnvironment, ExternalEnvironment
from src.area import Area
import xml.etree.ElementTree as ET


class BuildingFacilitySimulator:
    """シミュレータを表すオブジェクトで、外部プログラムとのやり取りを担当

    TODO: AI側からアクセスするときのメソッドを用意する（値取得、設定変更など）
    """

    areas: list[Area] = []
    ext_envs: list[ExternalEnvironment] = []
    area_envs: dict[str, list[AreaEnvironment]] = {}


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
        """2.6節のシミュレーションを1サイクル分進めるメソッド
        """

        for t, ext_env in enumerate(self.ext_envs):
            total_power_consumption = 0.
            for area_id, area in self.areas.items():
                area.update(ext_env, self.area_envs[area_id][t])
                total_power_consumption += area.power_consumption

            yield total_power_consumption
            
