from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element


T = TypeVar('T', bound='NamedTuple')

class ExternalEnvironment(NamedTuple):
    """2.4節(a) '全体の環境変数' を表すオブジェクト
    """

    solar_radiation: float
    temperature: float
    electric_price_unit: float

    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "value", f"invalid element for {cls}"

        return cls(
            solar_radiation=float(elem.attrib['solar-radiation']),
            temperature=float(elem.attrib['temperature']),
            electric_price_unit=float(elem.attrib['electric-price-unit'])
        )


class AreaEnvironment(NamedTuple):
    """2.4節(b) '各エリアごとの環境変数' を表すオブジェクト
    """

    people: int
    heat_source: float

    def calc_beta(self) -> float:
        """peopleとheat_sourceから2.3節の熱量betaを計算する
        """
        return self.people * 10 + self.heat_source


    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "value", f"invalid element for {cls}"

        return cls(
            people=int(elem.attrib['people']),
            heat_source=float(elem.attrib['heat-source'])
        )