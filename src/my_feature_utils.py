
import pandas as pd

# 1. すべての特徴量（英語キー）と日本語説明のマスター辞書
feature_master = {
    # === 元の特徴量（Original） ===
    "MedInc": "ブロック内における世帯所得の中央値",
    "HouseAge": "ブロック内における住宅築年数の中央値",
    "AveRooms": "1世帯あたりの平均部屋数（リビングやキッチン等を含む）",
    "AveBedrms": "1世帯あたりの平均寝室数",
    "Population": "ブロック内の総人口",
    "AveOccup": "1世帯あたりの平均同居人数",
    "Latitude": "ブロックの位置を示す北緯の度数",
    "Longitude": "ブロックの位置を示す西経の度数",
    "AllRooms": "ブロック内における全世帯の総部屋数の合計（新登場）",
    "AllBedrms": "ブロック内における全世帯の総部屋数の合計", 
    "Household": "ブロック内の総世帯数（新登場）",
    "Price": "目的変数：ブロックの住宅価格の中央値",

    # === 作成した特徴量（Engineered） ===
    # 基本統計・比率
    "room_per_household": "ブロック全体の総部屋数を世帯数で割った、1世帯あたりの部屋数",
    "bedroom_ratio": "総部屋数に対する寝室の割合（間取りの構造を示す指標）",
    "population_density": "総人口を世帯数で割った、1世帯あたりの平均人数に近い指標",
    
    # 主要都市（LA / SF / SD）への距離・方向
    "dist_to_la": "ロサンゼルス（LA）の中心地までの直線距離",
    "dist_to_sf": "サンフランシスコ（SF）の中心地までの直線距離",
    "dist_to_sd": "サンディエゴ（SD）の中心地までの直線距離",
    "dist_to_nearest_city": "主要3都市（LA / SF / SD）のうち、最も近い都市までの直線距離",
    "angle_to_la": "ロサンゼルス（LA）から見たブロックの位置する方角（ラジアン角度）",
    "angle_to_sf": "サンフランシスコ（SF）から見たブロックの位置する方角（ラジアン角度）",
    "angle_to_sd": "サンディエゴ（SD）から見たブロックの位置する方角（ラジアン角度）",
    
    # 空間の傾き・内陸度
    "lat_plus_lon": "緯度と経度の和（海岸線からの距離・内陸への深さを示す指標）",
    "lat_minus_lon": "緯度と経度の差（北東から南西方向への斜めの地理的傾きを示す指標）",
    
    # クラスタリング（エリア分け）
    "geo_cluster": "緯度・経度をもとにガウス混合モデル（GMM）で15分割したエリア番号",
    
    # 地域ごとの統計量・差分特徴量
    "MedInc_cluster_mean": "所属するGMMクラスター内における所得中央値の平均",
    "MedInc_diff_from_cluster_mean": "自分のブロックの所得が、地域の平均と比べてどれだけ高いか（差分）",
    "AveRooms_cluster_mean": "所属するGMMクラスター内における平均部屋数の平均",
    "AveRooms_diff_from_cluster_mean": "自分のブロックの部屋数が、地域の平均と比べてどれだけ広いか（差分）",
    "HouseAge_cluster_mean": "所属するGMMクラスター内における住宅築年数の中央値の平均",
    "HouseAge_diff_from_cluster_mean": "自分のブロックの築年数が、地域の平均と比べてどれだけ古い・新しいか（差分）"
}

# 2. 辞書を綺麗にテーブル成形して出力する関数
def print_feature_directory(df_columns=None):
    """特徴量辞書を綺麗に成形して出力する関数。"""
    data = []
    original_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", 
        "AveOccup", "Latitude", "Longitude", "AllRooms", "Household", "Price"
    ]
    for f_name, f_desc in feature_master.items():
        if df_columns is not None and f_name not in df_columns:
            continue
        category = "Original" if f_name in original_features else "Engineered"
        data.append({"Category": category, "Feature Name": f_name, "Description": f_desc})
    
    df_info = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', None)
    
    print("=" * 100)
    print(f" カリフォルニア住宅価格予測 : 特徴量ディレクトリ (全 {len(df_info)} 変数)")
    print("=" * 100)
    
    try:
        from IPython.display import display
        display(df_info.set_index(['Category', 'Feature Name']))
    except ImportError:
        print(df_info.to_string(index=False))
    print("=" * 100)

# 【ここを修正】このファイルを直接実行した時だけテスト出力し、importされた時は実行しない
if __name__ == "__main__":
    print_feature_directory()