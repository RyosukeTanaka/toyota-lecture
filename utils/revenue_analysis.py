# utils/revenue_analysis.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional, Union, Any

def calculate_daily_change(df: pd.DataFrame, cumulative_col: str) -> pd.Series:
    """日次変化量（新規予約数）を計算する補助関数
    
    Parameters:
    -----------
    df: pd.DataFrame
        累積値を含むデータフレーム（リードタイム降順でソート済み）
    cumulative_col: str
        累積値の列名
        
    Returns:
    --------
    pd.Series: 日次変化量。時間の流れに沿った新規予約数（正の値が増加）
    """
    if df.empty or cumulative_col not in df.columns:
        st.warning(f"日次変化量計算: データが空または列 '{cumulative_col}' が見つかりません。")
        return pd.Series(dtype='float64')
    
    # データがリードタイム降順（大→小）にソートされている前提
    # 時間の流れに沿った差分を計算するため、後の時点（リードタイム小）- 前の時点（リードタイム大）
    daily_change = df[cumulative_col].copy()
    
    # 初日（リードタイム最大）はそのままの値
    if len(daily_change) > 1:
        # 時間の流れに沿った差分計算（隣接する行の差分）
        # 注意: リードタイム降順なので、現在の値 - 一つ前（リードタイム大きい）の値
        original_values = daily_change.values.copy()
        for i in range(1, len(daily_change)):
            daily_change.iloc[i] = original_values[i] - original_values[i-1]
    
    return daily_change


def calculate_revenue_difference(
    df_actual: pd.DataFrame,
    df_predicted: pd.DataFrame,
    lead_time_col: str,
    actual_usage_col: str,
    pred_usage_col: str,
    price_col: str,
    change_lead_time: int
) -> Tuple[pd.DataFrame, float, float, float]:
    """実績と予測シナリオの売上差額を計算する関数
    
    Parameters:
    -----------
    df_actual: pd.DataFrame
        実績データ
    df_predicted: pd.DataFrame
        予測データ
    lead_time_col: str
        リードタイム列名
    actual_usage_col: str
        実績利用台数累積列名
    pred_usage_col: str
        予測利用台数累積列名
    price_col: str
        価格列名（価格_トヨタを想定）
    change_lead_time: int
        価格変更点のリードタイム値
        
    Returns:
    --------
    Tuple[pd.DataFrame, float, float, float]:
        - 売上計算結果のデータフレーム
        - 実績総売上
        - 予測総売上
        - 差額（実績 - 予測）
    """
    # 必要な列が揃っているか確認
    required_cols_actual = [lead_time_col, actual_usage_col, price_col]
    required_cols_predicted = [lead_time_col, pred_usage_col]
    
    missing_actual = [col for col in required_cols_actual if col not in df_actual.columns]
    missing_predicted = [col for col in required_cols_predicted if col not in df_predicted.columns]
    
    if missing_actual or missing_predicted:
        missing = list(set(missing_actual + missing_predicted))
        st.error(f"売上計算エラー: 必要な列 ('{', '.join(missing)}') がデータフレームに存在しません。")
        empty_df = pd.DataFrame()
        return empty_df, 0.0, 0.0, 0.0
    
    try:
        # 結果格納用のDataFrame
        result_df = pd.DataFrame()
        
        # リードタイムで並べ替え（降順: 大きい方から小さい方へ）
        df_actual_sorted = df_actual.sort_values(by=lead_time_col, ascending=False)
        df_pred_sorted = df_predicted.sort_values(by=lead_time_col, ascending=False)
        
        # 価格変更点以降のデータに絞る
        df_actual_filtered = df_actual_sorted[df_actual_sorted[lead_time_col] <= change_lead_time]
        df_pred_filtered = df_pred_sorted[df_pred_sorted[lead_time_col] <= change_lead_time]
        
        if df_actual_filtered.empty or df_pred_filtered.empty:
            st.warning(f"売上計算警告: LT {change_lead_time} 以降のデータが不足しています。")
            empty_df = pd.DataFrame()
            return empty_df, 0.0, 0.0, 0.0
        
        # 各リードタイムにおける新規予約数の計算（差分）
        result_df[lead_time_col] = df_actual_filtered[lead_time_col]
        
        # 日次変化量の計算（時間の流れに沿った差分）
        result_df['actual_new_bookings'] = calculate_daily_change(df_actual_filtered, actual_usage_col)
        result_df['pred_new_bookings'] = calculate_daily_change(df_pred_filtered, pred_usage_col)
        
        # 価格の取得
        result_df['actual_price'] = df_actual_filtered[price_col]
        
        # 予測シナリオでは、価格変更点の価格を固定使用
        fixed_price_row = df_actual_filtered[df_actual_filtered[lead_time_col] == change_lead_time]
        if not fixed_price_row.empty:
            fixed_price = fixed_price_row[price_col].iloc[0]
            result_df['fixed_price'] = fixed_price
        else:
            # 該当するリードタイムがない場合、最も近いリードタイムの価格を使用
            lt_diff = abs(df_actual_filtered[lead_time_col] - change_lead_time)
            closest_idx = lt_diff.idxmin()
            fixed_price = df_actual_filtered.loc[closest_idx, price_col]
            result_df['fixed_price'] = fixed_price
            st.warning(f"売上計算警告: LT {change_lead_time} の価格データがないため、最も近いLT {df_actual_filtered.loc[closest_idx, lead_time_col]} の価格 {fixed_price} を使用します。")
        
        # 売上の計算
        result_df['actual_revenue'] = result_df['actual_new_bookings'] * result_df['actual_price']
        result_df['predicted_revenue'] = result_df['pred_new_bookings'] * result_df['fixed_price']
        
        # 累計売上の計算
        result_df['actual_revenue_cumsum'] = result_df['actual_revenue'].cumsum()
        result_df['predicted_revenue_cumsum'] = result_df['predicted_revenue'].cumsum()
        
        # 差額の計算（実績 - 予測）
        result_df['revenue_difference'] = result_df['actual_revenue'] - result_df['predicted_revenue']
        result_df['revenue_difference_cumsum'] = result_df['actual_revenue_cumsum'] - result_df['predicted_revenue_cumsum']
        
        # 合計値の計算
        total_actual = result_df['actual_revenue'].sum()
        total_predicted = result_df['predicted_revenue'].sum()
        total_difference = total_actual - total_predicted
        
        # NaN値をゼロに変換
        result_df = result_df.fillna(0)
        
        return result_df, total_actual, total_predicted, total_difference
    
    except Exception as e:
        st.error(f"売上計算中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        empty_df = pd.DataFrame()
        return empty_df, 0.0, 0.0, 0.0


def plot_revenue_comparison(revenue_df: pd.DataFrame, lead_time_col: str, title: str = "売上金額比較") -> go.Figure:
    """売上比較のグラフを作成する関数
    
    Parameters:
    -----------
    revenue_df: pd.DataFrame
        売上計算結果のデータフレーム
    lead_time_col: str
        リードタイム列名
    title: str, optional
        グラフタイトル
        
    Returns:
    --------
    go.Figure: 売上比較グラフ
    """
    try:
        if revenue_df.empty or lead_time_col not in revenue_df.columns:
            st.warning("売上比較グラフ: データが不足しています。")
            return go.Figure()
        
        required_cols = ['actual_revenue_cumsum', 'predicted_revenue_cumsum', 'revenue_difference_cumsum']
        missing_cols = [col for col in required_cols if col not in revenue_df.columns]
        
        if missing_cols:
            st.warning(f"売上比較グラフ: 必要な列 {missing_cols} がありません。")
            return go.Figure()
        
        fig = go.Figure()
        
        # 累計実績売上
        fig.add_trace(go.Scatter(
            x=revenue_df[lead_time_col],
            y=revenue_df['actual_revenue_cumsum'],
            mode='lines+markers',
            name='実績累計売上',
            line=dict(color='rgba(0, 123, 255, 0.8)', width=2),
            marker=dict(size=6)
        ))
        
        # 累計予測売上
        fig.add_trace(go.Scatter(
            x=revenue_df[lead_time_col],
            y=revenue_df['predicted_revenue_cumsum'],
            mode='lines+markers',
            name='予測累計売上（価格固定）',
            line=dict(color='rgba(220, 53, 69, 0.8)', width=2),
            marker=dict(size=6)
        ))
        
        # 累計差額
        fig.add_trace(go.Scatter(
            x=revenue_df[lead_time_col],
            y=revenue_df['revenue_difference_cumsum'],
            mode='lines+markers',
            name='累計差額（実績-予測）',
            line=dict(color='rgba(40, 167, 69, 0.8)', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        # グラフレイアウト設定
        fig.update_layout(
            title=title,
            xaxis_title='リードタイム',
            yaxis_title='累計売上金額（円）',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified",
            xaxis_autorange='reversed'
        )
        
        return fig
    
    except Exception as e:
        st.error(f"売上比較グラフ作成中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return go.Figure() 