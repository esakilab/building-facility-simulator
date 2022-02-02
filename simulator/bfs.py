from __future__ import annotations
from copy import deepcopy
from datetime import timedelta, datetime
from typing import List, Optional
import os
import glob
import xml.etree.ElementTree as ET

from simulator.area import Area
from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.io import BuildingAction, BuildingState, Reward


class BuildingFacilitySimulator:
    """シミュレータを表すオブジェクトで、外部プログラムとのやり取りを担当
    """

    def __init__(self, cfg_path: str):
        self.cur_steps = 0
        self.areas = []
        self.ext_envs = []
        self.area_envs = {}
        
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

        # TODO: add `start_time` to the config file
        self.start_time = datetime.strptime(ext_env_elem[0].attrib["time"], "%Y-%m-%d %H:%M")

        self.total_steps = len(self.ext_envs)


    def get_area_env(self, area_id: int, timestamp: int):
        if area_id in self.area_envs:
            return self.area_envs[area_id][timestamp]
        else:
            return AreaEnvironment.empty()

    
    def has_finished(self):
        return self.cur_steps == self.total_steps

        
    def step(self, action: BuildingAction) -> tuple[BuildingState, Reward]:
        """2.6節のシミュレーションを1サイクル分進めるメソッド
        while not bfs.has_finished():
            for i in range(10):
                action = compute_action()
                (state, reward) = bfs.step(action)
            
            update_model()

        みたいにすると、10stepごとにモデルの更新を行える
        """
        if self.has_finished():
            return (None, None)

        ext_env = self.ext_envs[self.cur_steps]

        for area_id, area in enumerate(self.areas):
            area.update(action[area_id], ext_env, self.get_area_env(area_id, self.cur_steps))

        state = self.get_state()

        self.cur_steps += 1
        self.last_state = state

        return (
            state,
            Reward.from_state(state)
        )

    
    def get_current_datetime(self):
        return self.start_time + timedelta(minutes=self.cur_steps)


    def get_state(self) -> BuildingState:
        area_states = [area.get_state() for area in self.areas]
        return BuildingState.create(area_states, self.ext_envs[self.cur_steps])


    def print_cur_state(self):
        print(f"\niteration {self.cur_steps} ({self.get_current_datetime()})")
        if not self.has_finished():
            print(self.ext_envs[self.cur_steps])

        for aid, (area, st) in enumerate(zip(self.areas, self.last_state.areas)):
            print(f"area {aid}: temp={area.temperature:.2f}, power={st.power_consumption:.2f}, {area.facilities[0]}")

        print(f"total power consumption: {self.last_state.power_balance:.2f}", flush=True)
    

    def __add__(self, other: BuildingFacilitySimulator) -> BuildingFacilitySimulator:
        assert self.area_envs.keys() == other.area_envs.keys()
        result = deepcopy(self)

        result.total_steps += other.total_steps
        result.ext_envs += other.ext_envs
        for key in result.area_envs:
            result.area_envs[key] += other.area_envs[key]

        return result
    

    def __mul__(self, other: int) -> BuildingFacilitySimulator:
        result = deepcopy(self)

        result.total_steps *= other
        result.ext_envs *= other
        for key in result.area_envs:
            result.area_envs[key] *= other
        
        return result


class BFSList(list[BuildingFacilitySimulator]):
    def __init__(self, 
            xml_dir_path: Optional[str] = None,
            load_xml_num: Optional[int] = None,
            xml_pathes: list[str] = []):
        
        if xml_dir_path:
            xml_pathes.extend(glob.glob(os.path.join(xml_dir_path, '*.xml')))
        
        if load_xml_num == None:
            load_xml_num = len(xml_pathes)
        
        super().__init__()

        for xml_path in sorted(xml_pathes)[:load_xml_num]:
            print(f"Loading from {xml_path}")
            self.append(BuildingFacilitySimulator(xml_path))


    def step(self, actions: List[BuildingAction]) -> List[tuple[BuildingState, Reward]]:
        assert len(actions) == len(self), "len(actions) must be as same as the number of buildings"

        return [
            bfs.step(action) for action, bfs in zip(actions, self)
        ]
