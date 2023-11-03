import requests
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field


class StationDistanceCheckInput(BaseModel):
    """Input for stations distance check"""
    from_station: str = Field(...,
                              description="Name of start station")
    to_station: str = Field(...,
                            description="Name of destination station")


class StationDistanceTool(BaseTool):
    name = "get_station_distance"
    description = "Get the distance between two stations"
    api_key = "XbmxcBOfsMc0teKv47uUEc95uz1rQE5Dxkfvss8OILm7kKQH23Y/OKgj0C/YJuxK3BW5Et4i5FXmU6RYsE6jVg=="
    url = f"http://data.humetro.busan.kr/voc/api/open_api_distance.tnn?serviceKey={api_key}&act=json&scode="
    res = requests.get(url)

    def _run(self, station_name: str):
        pass

    def _arun(self, station_name: str):
        raise NotImplementedError("TrainDistanceTool does not support async ")
    args_schema: Optional[Type[BaseModel]] = StationDistanceCheckInput
