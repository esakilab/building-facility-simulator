import xml.etree.ElementTree as ET

from src.area import Area
from src.environment import AreaEnvironment, ExternalEnvironment
from src.io import BuildingAction, BuildingState, Reward


class BuildingFacilitySimulator:
    """シミュレータを表すオブジェクトで、外部プログラムとのやり取りを担当

    TODO: AI側からアクセスするときのメソッドを用意する（値取得、設定変更など）
    """

    steps: int = 0
    last_state: BuildingState
    areas: list[Area] = []
    ext_envs: list[ExternalEnvironment] = []
    area_envs: dict[int, list[AreaEnvironment]] = {}


    def __init__(self, cfg_path: str):
        root = ET.parse(cfg_path).getroot()
        
        assert root.tag == "BFS", "invalid BFS XML"

        area_elems = filter(lambda elem: elem.tag == 'area', root)
        area_env_elems = filter(lambda elem: elem.tag == 'area-environment', root)

        for area_elem in sorted(area_elems, key=lambda elem: elem.attrib['id']):
            assert int(area_elem.attrib['id']) == len(self.areas), \
                "Area IDs must start from 0 and must be consecutive."

            self.areas.append(Area.from_xml_element(area_elem))
        
        for area_env_elem in area_env_elems:
            area_id = int(area_env_elem.attrib['area-id'])
            self.area_envs[area_id] = [
                AreaEnvironment.from_xml_element(child) for child in area_env_elem
            ]
        
        ext_env_elem = next(filter(lambda elem: elem.tag == 'environment', root))

        self.ext_envs = [
            ExternalEnvironment.from_xml_element(child) for child in ext_env_elem
        ]


    def get_area_env(self, area_id: int, timestamp: int):
        if area_id in self.area_envs:
            return self.area_envs[area_id][timestamp]
        else:
            return AreaEnvironment.empty()

        
    def step(self, action: BuildingAction) -> tuple[BuildingState, Reward]:
        """2.6節のシミュレーションを1サイクル分進めるメソッド
        """

        for t, ext_env in enumerate(self.ext_envs):

            area_states = [
                area.update(action[area_id], ext_env, self.get_area_env(area_id, t))
                for area_id, area in enumerate(self.areas)
            ]

            state = BuildingState.create(area_states, ext_env.electric_price_unit)

            self.steps = t
            self.last_state = state

            yield (
                state,
                Reward.from_state(state)
            )


    def print_cur_state(self):
        print(f"\niteration {self.steps}")
        print(self.ext_envs[self.steps])

        for aid, (area, st) in enumerate(zip(self.areas, self.last_state.areas)):
            print(f"area {aid}: temp={area.temperature:.2f}, power={st.power_consumption:.2f}, {area.facilities[0]}")

        print(f"total power consumption: {self.last_state.power_balance:.2f}")
