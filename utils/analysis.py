# utils/analysis.py
import pandas as pd
import streamlit as st # エラー/情報表示用
from typing import Optional, Union, Dict, List # Dict と List を追加
import datetime

def analyze_unique_count_after_date(data: pd.DataFrame, start_date: datetime.date, booking_date_col: str, count_col: str) -> Optional[int]:
    """指定された日付以降の予約データで、指定列のユニーク数をカウントする"""
    if data is None or data.empty:
        st.warning("分析対象データがありません。")
        return None
    if booking_date_col not in data.columns or not pd.api.types.is_datetime64_any_dtype(data[booking_date_col]):
        st.error(f"'{booking_date_col}' 列が見つからないか、日付型ではありません。")
        return None
    if count_col not in data.columns:
        st.error(f"カウント対象列 '{count_col}' が見つかりません。")
        return None

    try:
        # 日付型に変換（比較のため）
        start_dt = pd.to_datetime(start_date)

        # 指定日より後のデータをフィルタリング
        data_after = data[data[booking_date_col] > start_dt].copy()

        if data_after.empty:
            st.info(f"{start_date} より後の予約日を持つデータが見つかりませんでした。")
            return 0 # データがない場合はカウント0

        # ユニーク数を計算 (NaNは除く)
        unique_count = data_after[count_col].nunique()
        return unique_count

    except Exception as e:
        st.error(f"利用台数カウント分析中にエラーが発生しました: {e}")
        return None

def analyze_daily_sum_after_date(data: pd.DataFrame, start_date: datetime.date, booking_date_col: str, sum_col: str) -> Optional[pd.DataFrame]:
    """指定された日付以降の予約データで、指定列の日毎の合計値を計算する"""
    if data is None or data.empty:
        st.warning("分析対象データがありません。")
        return None
    if booking_date_col not in data.columns or not pd.api.types.is_datetime64_any_dtype(data[booking_date_col]):
        st.error(f"'{booking_date_col}' 列が見つからないか、日付型ではありません。")
        return None
    if sum_col not in data.columns:
        st.error(f"集計対象列 '{sum_col}' が見つかりません。")
        return None
    # 集計対象列が数値型か確認 (NaNを許容)
    if not pd.api.types.is_numeric_dtype(data[sum_col]):
         # Try converting, coercing errors to NaN
         try:
             data[sum_col] = pd.to_numeric(data[sum_col], errors='coerce')
             if data[sum_col].isnull().all(): # If all values became NaN
                  st.error(f"集計対象列 '{sum_col}' は数値に変換できません。")
                  return None
             else:
                  st.warning(f"集計対象列 '{sum_col}' に数値でない値が含まれていたため、NaNとして扱います。")
         except Exception:
              st.error(f"集計対象列 '{sum_col}' は数値型である必要があります。")
              return None


    try:
        # 日付型に変換（比較のため）
        start_dt = pd.to_datetime(start_date)

        # 指定日より後のデータをフィルタリング & 必要な列を選択
        data_after = data.loc[data[booking_date_col] > start_dt, [booking_date_col, sum_col]].copy()

        if data_after.empty:
            st.info(f"{start_date} より後の予約日を持つデータが見つかりませんでした。")
            # 空のDataFrameを返す（グラフ描画でエラーにならないように）
            return pd.DataFrame(columns=[booking_date_col, f'{sum_col}_合計']).set_index(booking_date_col)

        # 予約日ごとに合計を計算 (NaNは0として扱わない場合は skipna=True (デフォルト))
        # groupby().sum() はデフォルトで数値列のみを対象とする
        daily_sum = data_after.groupby(data_after[booking_date_col].dt.date)[sum_col].sum()

        # 結果をDataFrameに変換して返す（グラフ描画用）
        daily_sum_df = daily_sum.reset_index()
        daily_sum_df.columns = [booking_date_col, f'{sum_col}_合計']
        # 日付列をdatetimeに戻してからインデックスに設定
        daily_sum_df[booking_date_col] = pd.to_datetime(daily_sum_df[booking_date_col])
        daily_sum_df = daily_sum_df.set_index(booking_date_col)

        return daily_sum_df

    except Exception as e:
        st.error(f"日毎の合計値分析中にエラーが発生しました: {e}")
        return None

def analyze_price_change_details_in_range(
    data: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    date_col: str,
    car_class_col: str,
    lead_time_col: str,
    price_cols: List[str]
) -> pd.DataFrame:
    """指定された日付範囲内の各利用日・車両クラスにおける価格変更点を詳細に分析する"""
    results = []
    if data is None or data.empty:
        st.warning("価格変動分析: 対象データがありません。")
        return pd.DataFrame(results)

    required_cols = [date_col, car_class_col, lead_time_col] + price_cols
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        st.error(f"価格変動分析エラー: 必要な列が見つかりません: {missing}")
        return pd.DataFrame(results)

    # 日付列がdatetime型であることを確認
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        try:
            data[date_col] = pd.to_datetime(data[date_col])
        except Exception as e_conv:
            st.error(f"価格変動分析エラー: 日付列 '{date_col}' を日付型に変換できません: {e_conv}")
            return pd.DataFrame(results)

    unique_dates_in_data = sorted(data[date_col].dt.date.unique())
    target_dates = [d for d in unique_dates_in_data if start_date <= d <= end_date]

    if not target_dates:
        st.info(f"指定された範囲 ({start_date} ～ {end_date}) に該当する利用日データがありませんでした。")
        return pd.DataFrame(results)

    # from .data_processing import find_last_price_change_lead_time # 遅延インポート (循環回避)
    # find_last_price_change_lead_time をここで直接利用できないため、ロジックを一部再実装または簡略化する
    # ここでは find_last_price_change_lead_time が呼び出し元でインポートされていることを期待するか、
    # もしくは、この関数を data_processing.py に移すことを検討。
    # 今回は呼び出し元 (page_analysis.py) で data_processing をインポートするため、そこで解決される。

    unique_car_classes = sorted(data[car_class_col].unique())

    for current_date in target_dates:
        for car_class in unique_car_classes:
            data_filtered = data[
                (data[date_col].dt.date == current_date) &
                (data[car_class_col] == car_class)
            ]
            if data_filtered.empty or lead_time_col not in data_filtered.columns:
                continue

            data_sorted = data_filtered.sort_values(by=lead_time_col, ascending=False).copy()
            
            # 各価格列に対して変更点をチェック
            for price_col_name in price_cols:
                if price_col_name not in data_sorted.columns:
                    continue
                
                # 価格変動があったかどうかの簡易チェック
                if data_sorted[price_col_name].nunique(dropna=True) <= 1:
                    continue # この価格列は変動なし
                
                # 前の行との価格差を計算 (リードタイム降順なので shift(1) は時間的に過去)
                data_sorted[f'{price_col_name}_prev'] = data_sorted[price_col_name].shift(1)
                # 変更があった行を特定 (NaNとの比較は常にFalseになるので、fillnaで対応するか、neで比較)
                # NAでない値同士の比較で変化があった行
                changes = data_sorted[data_sorted[price_col_name].notna() & \
                                      data_sorted[f'{price_col_name}_prev'].notna() & \
                                      (data_sorted[price_col_name] != data_sorted[f'{price_col_name}_prev'])]

                for idx, row in changes.iterrows():
                    # 変更があった行のリードタイムを取得
                    change_lt = row[lead_time_col]
                    price_after = row[price_col_name]
                    price_before = row[f'{price_col_name}_prev']
                    
                    results.append({
                        "利用日": current_date,
                        "車両クラス": car_class,
                        "価格列": price_col_name,
                        "変更リードタイム": change_lt,
                        "変更前価格": price_before,
                        "変更後価格": price_after,
                        "予約日(変更後価格適用開始)": pd.to_datetime(current_date) - pd.to_timedelta(change_lt, unit='D'),
                    })
    
    if not results:
        st.info(f"指定された範囲 ({start_date} ～ {end_date}) で価格変動は見つかりませんでした。")

    return pd.DataFrame(results).sort_values(by=["利用日", "車両クラス", "変更リードタイム", "価格列"]).reset_index(drop=True) 