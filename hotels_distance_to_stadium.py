import time
import math
import re
import urllib.parse
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# =====================
# 工具函数
# =====================

def init_driver():
    options = Options()
    options.add_argument("--headless=new")       # 无界面模式
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # 看情况加代理 / UA
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def extract_coords_from_url(url: str):
    """
    从 Google Maps 的 URL 中解析出坐标
    支持两种格式：
    1. 旧格式: .../@37.4020941,-121.968968,17z/...
    2. 新格式: ...!3d33.416023!4d-112.007295...
    """
    # 尝试旧格式
    m = re.search(r"@([\-0-9\.]+),([\-0-9\.]+),", url)
    if m:
        lat = float(m.group(1))
        lng = float(m.group(2))
        return lat, lng

    # 尝试新格式
    m = re.search(r"!3d([\-0-9\.]+)!4d([\-0-9\.]+)", url)
    if m:
        lat = float(m.group(1))
        lng = float(m.group(2))
        return lat, lng

    return None


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """haversine 公式，计算两点之间的球面距离（单位：米）"""
    R = 6371000  # 地球半径（米）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# =====================
# 1. 获取球场坐标（通过搜索球场）
# =====================

def get_stadium_coords(driver, stadium_name, state):
    """
    在 Google Maps 中搜索球场，解析 URL 拿到球场坐标
    """
    query = f"{stadium_name} {state}"
    url = "https://www.google.com/maps/search/" + urllib.parse.quote(query)
    driver.get(url)

    try:
        # 等待地图加载（左侧结果列表出现）
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='feed']"))
        )
    except Exception as e:
        print(f"  [get_stadium_coords] feed not found for {stadium_name}: {e}")
        # 有时候直接就是 place 页面，没有 feed，也可以直接用当前 URL
        pass

    time.sleep(2)  # 稍微等一下 URL 稳定

    current_url = driver.current_url
    coords = extract_coords_from_url(current_url)
    if coords:
        return coords

    print(f"  [get_stadium_coords] Cannot parse coords from URL: {current_url}")
    return None


# =====================
# 2. 找球场附近酒店，对每家酒店解析坐标并算距离
# =====================

def get_hotels_near_stadium(driver, stadium_name, state, stadium_coords,
                            max_hotels=10):
    """
    使用 Google Maps 搜索 `{stadium_name} {state} hotels`，
    抓取搜索列表中的酒店名、坐标，并计算到球场的直线距离。
    """
    lat, lng = stadium_coords
    query = f"hotels near {stadium_name} {state}"
    url = "https://www.google.com/maps/search/" + urllib.parse.quote(query)
    driver.get(url)

    # 等待搜索结果列表
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='feed']"))
        )
    except Exception as e:
        print(f"  [get_hotels_near_stadium] feed not found for {stadium_name}: {e}")
        return []

    time.sleep(2)

    hotels_data = []

    # Google Maps 结果列表里，通常每个结果是一个 <a> 标签，带 aria-label 和 href=/place/...
    # 这个选择器不一定 100% 稳，后续可以自己调
    cards = driver.find_elements(By.CSS_SELECTOR, "div[role='feed'] a[href*='/place/']")
    print(f"  Found {len(cards)} hotel candidates")
    cards = cards[:max_hotels]

    for card in cards:
        try:
            name = card.get_attribute("aria-label") or ""
            href = card.get_attribute("href") or ""
            # 解析酒店坐标
            hotel_coords = extract_coords_from_url(href)
            if not hotel_coords:
                continue
            hlat, hlng = hotel_coords
            distance_m = haversine_distance_m(lat, lng, hlat, hlng)

            hotels_data.append({
                "hotel_name": name,
                "hotel_url": href,
                "hotel_lat": hlat,
                "hotel_lng": hlng,
                "distance_m": distance_m
                # "可承载人数"这里暂时没法自动拿到，只能后续手动或爬官网
            })
        except Exception as e:
            print("    error parsing hotel card:", e)
            continue

    return hotels_data


# =====================
# 3. 主程序：循环所有 NFL 球场
# =====================

def main():
    input_csv = "nfl_stadium_capacity.csv"       # 你前面生成的那个
    output_csv = "nfl_stadium_hotels_from_gmaps.csv"

    driver = init_driver()

    df = pd.read_csv(input_csv)
    all_rows = []

    for idx, row in df.iterrows():
        team = row["Team"]
        stadium = row["Stadium"]
        state = row["State"]

        print(f"\n=== [{idx+1}/{len(df)}] {team} - {stadium} ({state}) ===")

        # 1) 获取球场坐标
        coords = get_stadium_coords(driver, stadium, state)
        if not coords:
            print("  -> skip this stadium (no coords)")
            continue
        s_lat, s_lng = coords
        print(f"  Stadium coords: {s_lat}, {s_lng}")

        # 2) 获取附近酒店
        hotels = get_hotels_near_stadium(driver, stadium, state, coords, max_hotels=10)
        print(f"  Valid hotels: {len(hotels)}")

        for h in hotels:
            all_rows.append({
                "team": team,
                "stadium": stadium,
                "state": state,
                "stadium_lat": s_lat,
                "stadium_lng": s_lng,
                "hotel_name": h["hotel_name"],
                "hotel_url": h["hotel_url"],
                "hotel_lat": h["hotel_lat"],
                "hotel_lng": h["hotel_lng"],
                "distance_m": h["distance_m"],
                # capacity 这里先留空，后面你可以手动补
                "capacity": None
            })

        # 简单防封：每个球场之间 sleep 一下
        time.sleep(3)

    driver.quit()

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print("\nDone! Saved to", output_csv)


if __name__ == "__main__":
    main()
