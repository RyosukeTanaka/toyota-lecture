# utils/analysis.py
import pandas as pd
import streamlit as st # エラー/情報表示用
from typing import Optional, Union, Dict # Dict を追加
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