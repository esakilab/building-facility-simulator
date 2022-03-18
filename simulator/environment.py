from typing import Optional, Type, TypeVar

from pydantic import BaseModel


E = TypeVar('E', bound='ExternalEnvironment')

class ExternalEnvironment(BaseModel):
    """2.4節(a) '全体の環境変数' を表すオブジェクト
    """

    solar_radiation:     float # [W/m^2]
    temperature:         float # [℃]
    electric_price_unit: float # [¥/kWh]


A = TypeVar('A', bound='AreaEnvironment')

class AreaEnvironment(BaseModel):
    """2.4節(b) '各エリアごとの環境変数' を表すオブジェクト
    """

    people: int
    heat_source: float # [W]

    def calc_beta(self) -> float:
        """peopleとheat_sourceから、2.3節の熱量betaのうち、環境による部分を計算する
        """
        return self.heat_source * 60 / 1000


class BuildingEnvironment(BaseModel):
    external: ExternalEnvironment
    areas: list[Optional[AreaEnvironment]]
