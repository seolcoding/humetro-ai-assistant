import os
import requests
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import sys
sys.path.append('../..')
load_dotenv(find_dotenv())


class TrainScheduleCheckInput(BaseModel):
    """Input for Train Schedule Check"""
    station_name: str = Field(..., description="Name of the station to check")


class TrainScheduleTool(BaseTool):
    name = "get_train_schedule"
    description = "Get the train schedule for a given station"

    params = {
        "api_key": os.environ.get("TRAIN_PORTAL_API_KEY"),
        "day_code": "8",
        "line_code":  "1",
        "station_code": "102",
    }

    # endpoint = f'https://openapi.kric.go.kr/openapi/trainUseInfo/subwayTimetable?serviceKey={api_key}&format=json&railOprIsttCd=BS&dayCd={day_code}&lnCd={line_code}&stinCd={station_code}'

    def _run(self, station_name: str):
        res = requests.get(endpoint)
        schedule = res.json()['body']
        schedule.sort(key=lambda x: x['arvTm'])

        for s in schedule[:10]:
            print(s['arvTm'], s['trnNo'])

        for s in schedule[-10:]:
            print(s['arvTm'], s['trnNo'])
            pass

    def _arun(self, station_name: str):
        raise NotImplementedError("TrainScheduleTool does not support async ")
    args_schema: Optional[Type[BaseModel]] = TrainScheduleCheckInput


if __name__ == "__main__":

    params = {
        "serviceKey": os.environ.get("TRAIN_PORTAL_API_KEY"),
        "dayCd": "8",
        "lnCd":  "1",
        "stinCd": "102",
        "format": "json",
        "railOprIsttCd": "BS",
    }
    endpoint = "https://openapi.kric.go.kr/openapi/trainUseInfo/subwayTimetable"
    res = requests.get(endpoint, params=params)
    schedule = res.json()['body']
    schedule.sort(key=lambda x: x['arvTm'])

    for s in schedule[:10]:
        print(s['arvTm'], s['trnNo'])

    for s in schedule[-10:]:
        print(s['arvTm'], s['trnNo'])
        pass
