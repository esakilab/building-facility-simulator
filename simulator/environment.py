from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element

from pydantic import BaseModel


E = TypeVar('E', bound='ExternalEnvironment')

class ExternalEnvironment(BaseModel):
    """2.4節(a) '全体の環境変数' を表すオブジェクト
    """

    solar_radiation:     float # [W/m^2]
    temperature:         float # [℃]
    electric_price_unit: float # [¥/kWh]
    
    @classmethod
    def from_xml_element(cls: Type[E], elem: Element) -> E:
        assert elem.tag == "value", f"invalid element for {cls}"

        return cls(
            solar_radiation=float(elem.attrib['solar-radiation']),
            temperature=float(elem.attrib['temperature']),
            electric_price_unit=float(elem.attrib['electric-price-unit'])
        )


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


    @classmethod
    def from_xml_element(cls: Type[A], elem: Element) -> A:
        assert elem.tag == "value", f"invalid element for {cls}"

        return cls(
            people=int(elem.attrib['people']),
            heat_source=float(elem.attrib['heat-source'])
        )
    
    @classmethod
    def empty(cls: Type[A]) -> A:

        return cls(
            people=0,
            heat_source=0
        )