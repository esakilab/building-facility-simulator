from time import sleep
from pprint import PrettyPrinter
from shutil import get_terminal_size
from src.area import Area
from src.bfs import BuildingFacilitySimulator

pp = PrettyPrinter(indent=2, width=get_terminal_size().columns)


def print_area(area_id: str, area: Area):
    print(f"area {area_id}: temp={area.temperature:.2f}, power={area.power_consumption:.2f}")


if __name__ == "__main__":
    bfs = BuildingFacilitySimulator("sample_cfg.xml")

    print("initial state")
    for area_id, area in bfs.areas.items():
        print_area(area_id, area)
    
    # pp.pprint(bfs.areas)
    # pp.pprint(bfs.ext_envs)
    # pp.pprint(bfs.area_envs)
    # print("-" * get_terminal_size().columns)

    for i, total_power in enumerate(bfs.next_step()):
        sleep(1)
        
        print(f"\n{i}th iteration")
        for area_id, area in bfs.areas.items():
            print_area(area_id, area)
        print(f"total power consumption: {total_power:.2f}")