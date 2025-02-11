import googlemaps
from datetime import datetime
from typing import Tuple, List, Dict, Optional


class GoogleMapsService:
    def __init__(self, api_key: str):
        """
        初始化Google Maps客户端

        Args:
            api_key: Google Maps API密钥
        """
        self.gmaps = googlemaps.Client(key=api_key)

    def geocode_address(self, address: str) -> List[Dict]:
        """
        将地址转换为地理坐标

        Args:
            address: 需要转换的地址字符串

        Returns:
            包含地理信息的字典列表
        """
        try:
            return self.gmaps.geocode(address)
        except Exception as e:
            raise Exception(f"地址解析失败: {str(e)}")

    def reverse_geocode(self, coordinates: Tuple[float, float],
                        enable_address_descriptor: bool = False) -> List[Dict]:
        """
        将坐标转换为地址

        Args:
            coordinates: (纬度, 经度)的元组
            enable_address_descriptor: 是否启用地址描述符

        Returns:
            包含地址信息的字典列表
        """
        try:
            return self.gmaps.reverse_geocode(
                coordinates,
                enable_address_descriptor=enable_address_descriptor
            )
        except Exception as e:
            raise Exception(f"反向地址解析失败: {str(e)}")

    def get_directions(self,
                       origin: str,
                       destination: str,
                       mode: str = "transit",
                       departure_time: Optional[datetime] = None,
                       arrival_time: Optional[datetime] = None,
                       waypoints: Optional[List[str]] = None,
                       alternatives: bool = False,
                       avoid: Optional[List[str]] = None) -> List[Dict]:
        """
        获取路线规划

        Args:
            origin: 起点地址
            destination: 终点地址
            mode: 交通方式 ("driving", "walking", "bicycling", "transit")
            departure_time: 出发时间
            arrival_time: 到达时间
            waypoints: 途经点列表
            alternatives: 是否返回多条路线
            avoid: 需要避开的路段类型 ["tolls", "highways", "ferries"]

        Returns:
            包含路线信息的字典列表
        """
        try:
            if departure_time is None:
                departure_time = datetime.now()

            return self.gmaps.directions(
                origin,
                destination,
                mode=mode,
                departure_time=departure_time,
                arrival_time=arrival_time,
                waypoints=waypoints,
                alternatives=alternatives,
                avoid=avoid
            )
        except Exception as e:
            raise Exception(f"路线规划失败: {str(e)}")

    def validate_address(self,
                         address: str,
                         region_code: str,
                         locality: Optional[str] = None,
                         enable_usps_cass: bool = True) -> Dict:
        """
        验证地址有效性

        Args:
            address: 需要验证的地址
            region_code: 国家/地区代码
            locality: 所在地区
            enable_usps_cass: 是否启用USPS CASS验证（仅限美国地址）

        Returns:
            地址验证结果
        """
        try:
            return self.gmaps.addressvalidation(
                [address],
                regionCode=region_code,
                locality=locality,
                enableUspsCass=enable_usps_cass
            )
        except Exception as e:
            raise Exception(f"地址验证失败: {str(e)}")

    def get_place_details(self,
                          place_id: str,
                          fields: Optional[List[str]] = None) -> Dict:
        """
        获取地点详细信息

        Args:
            place_id: 地点ID
            fields: 需要返回的字段列表

        Returns:
            地点详细信息
        """
        try:
            return self.gmaps.place(
                place_id,
                fields=fields
            )
        except Exception as e:
            raise Exception(f"获取地点信息失败: {str(e)}")

    def search_places_nearby(self,
                             location: Tuple[float, float],
                             radius: int = 1000,
                             keyword: Optional[str] = None,
                             type: Optional[str] = None) -> Dict:
        """
        搜索附近地点

        Args:
            location: (纬度, 经度)的元组
            radius: 搜索半径（米）
            keyword: 搜索关键词
            type: 地点类型

        Returns:
            附近地点列表
        """
        try:
            return self.gmaps.places_nearby(
                location=location,
                radius=radius,
                keyword=keyword,
                type=type
            )
        except Exception as e:
            raise Exception(f"搜索附近地点失败: {str(e)}")


# 使用示例：
def example_usage():
    # 初始化服务
    maps_service = GoogleMapsService('AIzaSyAK1xPLQZPA92PBQTaOCZHojpYE_ZJnRy4')

    # 地址解析示例
    try:
        geocode_result = maps_service.geocode_address('1600 Amphitheatre Parkway, Mountain View, CA')
        print("地址解析结果:", geocode_result)

        # 获取经纬度坐标
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            lat, lng = location['lat'], location['lng']

            # 搜索附近景点
            nearby_places = maps_service.search_places_nearby(
                location=(lat, lng),
                radius=1000,
                type='tourist_attraction'
            )
            print("附近景点:", nearby_places)

            # 获取路线规划
            directions = maps_service.get_directions(
                origin='Sydney Town Hall',
                destination='Parramatta, NSW',
                mode='transit'
            )
            print("路线规划:", directions)

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    example_usage()