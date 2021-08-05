from dataclasses import dataclass
from collections import defaultdict


class FacilityAction(dict):
    pass


class AreaAction(defaultdict):
    def __init__(self):
        super().__init__(FacilityAction)


class BuildingAction(defaultdict):
    """AIからの指示を表すオブジェクト
    """

    def __init__(self):
        super().__init__(AreaAction)
    

    def add(self, area_id: int, facility_id: int, **kwargs):
        self[area_id][facility_id].update(kwargs)