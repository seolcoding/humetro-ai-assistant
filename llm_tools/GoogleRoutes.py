from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool

import json
import requests
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


class GoogleRouteToolInputChecker(BaseModel):
    """Input checker for GoogleRoute Tool"""
    destination: str = Field(
        ..., description="Destination to get directions from 하단역")


class GoogleRouteTool(BaseTool):
    name = "GoogleRouteTool"
    description = "Get directions from 하단역 to a given destination Always use this tool when answering questions about directions"
    # TODO: 의존성 주입을 통해 다른 역에서도 출발할 수 있도록 하기
    # 현재는 하단 기준으로 하드코딩되어 있음.

    def call_api(self, destination: str) -> str:
        endpoint = 'https://routes.googleapis.com/directions/v2:computeRoutes'

        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': os.environ['GOOGLE_MAPS_KEY'],
            'X-Goog-FieldMask': 'routes.legs.steps.transitDetails',
        }

        payload = {
            "origin": {
                "address": "부산광역시 사하구 낙동남로 지하1415 (하단동)"
            },
            "destination": {
                "address": destination
            },
            "travelMode": "TRANSIT",
            "computeAlternativeRoutes": False,
            "transitPreferences": {
                "routingPreference": "LESS_WALKING",
                "allowedTravelModes": ["SUBWAY"]
            },
            "languageCode": "ko-KR",
            "units": "METRIC"
        }
        response = requests.post(endpoint, headers=headers,
                                 data=json.dumps(payload))
        return response

    def _run(self, destination: str) -> str:
        response = self.call_api(destination)
        result_prefix = "<<HUMETRO_AI_DIRECTIONS>>\n"
        if response.status_code != 200:
            try:
                error_msg = response.json()['error']['message']
                return result_prefix + "ERROR : " + error_msg
            except:
                return result_prefix + "ERROR : " + "알 수 없는 오류가 발생했습니다."
        if 'routes' not in response.json():
            return result_prefix + "ERROR : " + "해당 경로를 찾지 못했습니다."
        recommened_route = response.json()['routes'][0]['legs'][0]

        return result_prefix + self.parse_route_to_string(destination, recommened_route)

    def parse_route_to_string(self, destination: str, recommened_route: dict) -> str:
        steps = recommened_route['steps']
        order_num = 1
        result = f"**하단역에서 {destination}까지 이동경로**\n"
        for step in steps:
            if not step:
                continue
            # 부산 1호선
            transit_type = step['transitDetails']['transitLine']['vehicle']['type']
            transit_line = step['transitDetails']['transitLine']['nameShort']
            departure_stop = step['transitDetails']['stopDetails']['departureStop']['name']
            arrival_stop = step['transitDetails']['stopDetails']['arrivalStop']['name']
            stop_count = step['transitDetails']['stopCount']
            stop_placeholder = ''
            if transit_type == 'SUBWAY':
                departure_stop = self.add_station_suffix(departure_stop)
                arrival_stop = self.add_station_suffix(arrival_stop)
                stop_placeholder = '역'
                transit_line += '으'
            if transit_type == 'BUS':
                transit_line += '번 버스'
                departure_stop += '버스정류장'
                arrival_stop += '버스정류장'
                stop_placeholder = '정류장'
            result += f"{order_num}. {departure_stop} 에서 {arrival_stop}까지 {transit_line}로 {stop_count}개 {stop_placeholder}을 이동합니다.\n"
            order_num += 1
        result += f"{order_num}. {arrival_stop}에서 {destination}까지 도보로 이동합니다.\n"

        return result

    def add_station_suffix(self, station: str) -> str:
        station = station.replace('경찰서', '')

        if station[-1] == '역':
            return station
        else:
            return station + '역'

    args_schema: Optional[Type[BaseModel]] = GoogleRouteToolInputChecker


if __name__ == "__main__":
    busan_tourist_spots = [
        "영화의거리",
        "태종대",
        "감천문화마을",
        "부산타워",
        "용궁사",
        "부산아쿠아리움",
        "동백섬",
        "영도다리",
        "자갈치시장",
        "부산시립미술관",
        "송도해수욕장",
        "을숙도",
        "용두산공원",
        "부산영화의전당",
        "김해롯데워터파크",
        "이기대공원",
        "남포동",
        "흰여울문화마을",
        "해운대 해수욕장",
        "광안리",
    ]

    for loc in busan_tourist_spots:
        tool = GoogleRouteTool()
        result = tool._run(loc)
        print(result)
