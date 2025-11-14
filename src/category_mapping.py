"""指标分类映射模块

根据用户提供的分类标准，将指标名称映射到对应的分类。
"""

from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


# 定义分类及其关键词
CATEGORY_KEYWORDS = {
    "能源消费类": {
        "keywords": [
            "Natural gas total consumption",
            "Geothermal total consumption",
            "Wind energy total consumption",
            "Solar energy total consumption",
            "Hydropower total consumption",
            "Biomass total consumption",
            "All petroleum products total consumption",
            "Electricity total consumption",
            "Total energy consumption in the industrial sector",
            "Total energy consumption in the transportation sector",
            "Total energy consumption in the commercial sector",
            "Total energy consumption in the residential sector",
            "Total energy consumption per capita",
            "Total energy consumption per capita in the industrial sector",
            "Total energy consumption per capita in the transportation sector",
            "Total energy consumption per capita in the commercial sector",
            "Total energy consumption per capita in the residential sector",
            "Renewable energy total consumption",
            "Petroleum total consumption",
            "Coal total consumption",
            "Wood and Waste energy production",
            "Noncombustible renewable energy production",
            "Biofuels production",
            "Biodiesel production",
            "Fuel ethanol production",
            "Renewable diesel production",
            "Crude oil production",
            "Nuclear electricity net generation",
            "Nuclear energy consumed for electricity generation",
            "Renewable energy production",
            "Natural gas marketed production",
            "Coal production",
            "Total energy consumption",
            "Total energy production",
            "Natural gas total consumption (excluding supplemental gaseous fuels)",
        ],
        "exclude_keywords": ["price", "expenditure", "CO2", "emission", "total CO2"],
    },
    "能源价格类": {
        "keywords": [
            "Natural gas average price",
            "Total energy average price",
            "Total energy average price in the industrial sector",
            "Total energy average price in the transportation sector",
            "Total energy average price in the commercial sector",
            "Total energy average price in the residential sector",
            "All petroleum products average price",
            "Motor gasoline average price",
            "Electricity average price",
            "Coal average price",
            "Total energy expenditures",
            "Total energy expenditures in the industrial sector",
            "Total energy expenditures in the transportation sector",
            "Total energy expenditures in the commercial sector",
            "Total energy expenditures in the residential sector",
            "Total energy expenditures per capita",
            "All petroleum products total expenditures",
            "Motor gasoline total expenditures",
            "Natural gas total expenditures",
            "Energy expenditures as percent of current-dollar gross domestic product",
            "Coal total expenditures",
            "Electricity total expenditures",
            "Motor gasoline expenditures per capita",
            "Total energy total expenditures",
        ],
        "exclude_keywords": [],
    },
    "碳排放类": {
        "keywords": [
            "Total energy CO2 emissions",
            "Total energy CO2 emissions for the industrial sector",
            "Total energy CO2 emissions for the transportation sector",
            "Total energy CO2 emissions for the commercial sector",
            "Total energy CO2 emissions for the residential sector",
            "Per capita energy-related CO2 emissions",
            "Carbon intensity of energy supply",
            "Carbon intensity of economy",
            "Coal total CO2 emissions",
            "total CO2 emissions",
            "CO2 emissions",
        ],
        "exclude_keywords": [],
    },
    "经济人口类": {
        "keywords": [
            "Current-dollar gross domestic product",
            "Real gross domestic product",
            "Total energy consumption per dollar of real gross domestic product",
            "Resident population including armed forces",
        ],
        "exclude_keywords": [],
    },
    "气候环境类": {
        "keywords": [
            "Heating degree days",
            "Cooling degree days",
            "Electricity consumed by (sales to ultimate customers in) the residential sector per capita",
            "Electricity consumed by (sales to ultimate customers in) the residential sector",
        ],
        "exclude_keywords": [],
    },
    "城市水资源类": {
        "keywords": [
            "water withdrawals",
            "irrigation",
            "public supply",
            "thermoelectric",
            "industrial water",
        ],
        "exclude_keywords": [],
    },
    "航空出行类": {
        "keywords": [
            "enplanement",
            "airport",
            "air travel",
            "airports",
        ],
        "exclude_keywords": [],
    },
    "废弃物管理类": {
        "keywords": [
            "waste",
            "recycling",
            "composting",
        ],
        "exclude_keywords": [],
    },
    "比赛因素类": {
        "keywords": [
            "sports-factors-nfl-stadium-capacity",
            "sports-factors-stadium-hotel-distance-avg",
            "sports-factors-peak-public-transport-capacity",
        ],
        "exclude_keywords": [],
    },
}

INDICATOR_CATEGORY_OVERRIDES = {
    "city-water-use-cbsa-ps-wtotl": "城市水资源类",
    "city-water-use-cbsa-ir-irtot": "城市水资源类",
    "city-water-use-cbsa-to-wtotl": "城市水资源类",
    "city-water-use-cbsa-pt-wtotl": "城市水资源类",
    "city-water-use-cbsa-in-wtotl": "城市水资源类",
    "airtravel-cy2023-cy23-enplanements": "航空出行类",
    "airtravel-cy2023-cy22-enplanements": "航空出行类",
    "airtravel-cy2023-enplanement-growth": "航空出行类",
    "airtravel-cy2023-airport-count": "航空出行类",
    "sports-factors-nfl-stadium-capacity": "比赛因素类",
    "sports-factors-stadium-hotel-distance-avg": "比赛因素类",
    "sports-factors-peak-public-transport-capacity": "比赛因素类",
}


def classify_indicator(indicator_name: str) -> Optional[str]:
    """根据指标名称将其分类到对应的类别。
    
    Args:
        indicator_name: 指标名称
        
    Returns:
        分类名称，如果无法匹配则返回None
    """
    if not isinstance(indicator_name, str):
        return None
    
    indicator_name_lower = indicator_name.lower()
    
    # 优先检查碳排放类（因为可能与其他类别重叠）
    carbon_keywords = CATEGORY_KEYWORDS["碳排放类"]["keywords"]
    if any(keyword.lower() in indicator_name_lower for keyword in carbon_keywords):
        return "碳排放类"
    
    # 按优先级顺序检查其他分类
    for category, config in CATEGORY_KEYWORDS.items():
        if category == "碳排放类":
            continue  # 已经检查过了
        
        keywords = config["keywords"]
        exclude_keywords = config.get("exclude_keywords", [])
        
        # 检查是否包含排除关键词
        if any(exclude in indicator_name_lower for exclude in exclude_keywords):
            continue
        
        # 检查是否包含该分类的关键词
        for keyword in keywords:
            if keyword.lower() in indicator_name_lower:
                return category
    
    return None


def classify_indicators(
    indicator_names: Dict[str, str],
    metadata: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """批量分类指标。
    
    Args:
        indicator_names: 字典，key为indicator_id，value为indicator_name
        metadata: 可选的元数据DataFrame，包含indicator_id和unit列
        
    Returns:
        字典，key为indicator_id，value为分类名称
    """
    classification = {}
    for indicator_id, indicator_name in indicator_names.items():
        if indicator_id in INDICATOR_CATEGORY_OVERRIDES:
            classification[indicator_id] = INDICATOR_CATEGORY_OVERRIDES[indicator_id]
            continue
        category = classify_indicator(indicator_name)
        
        # 如果无法通过名称分类，尝试使用元数据中的单位信息
        if category is None and metadata is not None:
            meta_rows = metadata[metadata["indicator_id"] == indicator_id]
            if not meta_rows.empty:
                # 优先检查CO2相关信息（因为CO2指标更重要）
                has_co2 = False
                has_consumption = False
                
                for _, meta_row in meta_rows.iterrows():
                    unit = str(meta_row.get("unit", ""))
                    unit_lower = unit.lower()
                    indicator_name_from_meta = str(meta_row.get("indicator_name", ""))
                    name_lower = indicator_name_from_meta.lower()
                    
                    # 检查单位或指标名称中是否包含CO2相关信息
                    if ("co2" in unit_lower or "carbon" in unit_lower or 
                        "co2" in name_lower or "carbon" in name_lower):
                        has_co2 = True
                        break
                    # 检查是否包含消费相关关键词
                    if ("consumption" in name_lower or "production" in name_lower):
                        if "co2" not in unit_lower and "carbon" not in unit_lower:
                            has_consumption = True
                
                # 优先分配CO2类别
                if has_co2:
                    category = "碳排放类"
                elif has_consumption:
                    category = "能源消费类"
        
        classification[indicator_id] = category
    return classification


def get_category_indicators(
    classification: Dict[str, str], category: str
) -> List[str]:
    """获取指定分类下的所有指标ID。
    
    Args:
        classification: 指标分类字典
        category: 分类名称
        
    Returns:
        该分类下的指标ID列表
    """
    return [
        indicator_id
        for indicator_id, cat in classification.items()
        if cat == category
    ]
