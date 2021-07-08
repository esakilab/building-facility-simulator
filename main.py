from time import sleep

from src.area import Area
from src.bfs import BuildingFacilitySimulator
from src.io import AreaState, BuildingAction


def print_area(area_id: str, area: Area, area_state: AreaState):
    print(f"area {area_id}: temp={area.temperature:.2f}, power={area_state.power_consumption:.2f}, {area.facilities[0]}")


if __name__ == "__main__":
    bfs = BuildingFacilitySimulator("BFS_environment.xml")


    action = BuildingAction()
    action.add(area_id=1, facility_id=0, status=True, temperature=22)
    action.add(area_id=2, facility_id=0, status=True, temperature=25)
    action.add(area_id=3, facility_id=0, status=True, temperature=28)
    action.add(area_id=4, facility_id=0, mode="charge")

    for i, (building_state, reward) in enumerate(bfs.step(action)):
        sleep(0.1)
        
        print(f"\niteration {i}")
        print(bfs.ext_envs[i])
        for area_id, area in enumerate(bfs.areas):
            print_area(area_id, area, building_state.areas[area_id])
        
        # print(f"es ratio: {building_state.areas[4].facilities[0].charge_ratio}")
        print(f"total power consumption: {building_state.power_balance:.2f}")
