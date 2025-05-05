import pandas as pd
import numpy as np
import streamlit as st
import datetime
from typing import Optional, List, Tuple

def nullify_usage_data_after_date(df: pd.DataFrame, cutoff_date: datetime.date, date_col: str, cols_to_nullify: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[List[str]]]:
    """指定された日付以降のデータで、指定された列の値をNaNに更新し、更新後のDF、更新行数、更新列リストを返す"""
    if df is None or df.empty:
        st.warning("データ更新: 対象データがありません。")
        return df, None, None
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        st.error(f"データ更新エラー: 日付列 '{date_col}' が見つからないか、日付型ではありません。")
        return df, None, None

    original_cols_to_nullify = cols_to_nullify[:]
    valid_cols_to_nullify = [col for col in cols_to_nullify if col in df.columns]
    missing_cols = list(set(original_cols_to_nullify) - set(valid_cols_to_nullify))

    if missing_cols:
        st.warning(f"データ更新警告: NaNに設定する列が見つかりません: {missing_cols}")
        if not valid_cols_to_nullify:
             st.error("データ更新エラー: NaNに設定する有効な列がありません。")
             return df, None, None

    try:
        df_modified = df.copy()
        cutoff_dt = pd.to_datetime(cutoff_date)
        mask = df_modified[date_col] >= cutoff_dt
        num_updated = mask.sum()

        if num_updated > 0:
            for col in valid_cols_to_nullify:
                df_modified.loc[mask, col] = pd.NA
        # else: # データがない場合の情報は呼び出し元で表示
        #     st.info(f"データ更新: {cutoff_date} 以降に該当するデータはありませんでした。")

        return df_modified, num_updated, valid_cols_to_nullify

    except Exception as e:
        st.error(f"データ更新中にエラーが発生しました: {e}")
        return df, None, None

    return df_modified, num_updated, valid_cols_to_nullify 