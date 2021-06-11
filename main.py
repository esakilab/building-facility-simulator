from time import sleep
from src.area import Area
from src.bfs import BuildingFacilitySimulator


def print_area(area_id: str, area: Area):
    print(f"area {area_id}: temp={area.temperature:.2f}, power={area.power_consumption:.2f}, {area.facilities[0]}")


if __name__ == "__main__":
    bfs = BuildingFacilitySimulator("BFS_environment.xml")

    print("initial state")
    for area_id, area in enumerate(bfs.areas):
        print_area(area_id, area)
    
    bfs.areas[1].facilities[0].turn_on()
    bfs.areas[1].facilities[0].set_temperature(22)

    bfs.areas[2].facilities[0].turn_on()
    bfs.areas[2].facilities[0].set_temperature(25)

    bfs.areas[3].facilities[0].turn_on()
    bfs.areas[3].facilities[0].set_temperature(28)

    bfs.areas[4].facilities[0].start_charging()

    for i, total_power in enumerate(bfs.next_step()):
        sleep(0.1)
        
        print(f"\niteration {i}")
        print(bfs.ext_envs[i])
        for area_id, area in enumerate(bfs.areas):
            print_area(area_id, area)
        print(f"total power consumption: {total_power:.2f}")

        # if i % 9 == 0:
        #     if i % 2 == 0:
