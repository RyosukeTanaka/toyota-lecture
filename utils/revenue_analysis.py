# utils/revenue_analysis.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional, Union, Any

def calculate_revenue_difference(
    df_actual: pd.DataFrame,
    df_predicted: pd.DataFrame,
    lead_time_col: str,
    actual_usage_col: str,
    pred_usage_col: str,
    price_col: str,
    change_lead_time: int
) -> Tuple[pd.DataFrame, float, float, float, float, float, float]:
    """実績と予測シナリオの売上差額等を計算 (change_lead_time > 0)
       予測売上 = 変更前実績 + 変更後予測(旧価格)
       Returns: revenue_df, total_actual, total_predicted_hybrid, total_difference,
                actual_before, actual_after, predicted_after
    """
    # 必要な列が揃っているか確認
    required_cols_actual = [lead_time_col, actual_usage_col, price_col]
    required_cols_predicted = [lead_time_col, pred_usage_col]
    
    missing_actual = [col for col in required_cols_actual if col not in df_actual.columns]
    missing_predicted = [col for col in required_cols_predicted if col not in df_predicted.columns]
    
    if missing_actual or missing_predicted:
        missing = list(set(missing_actual + missing_predicted))
        st.error(f"売上計算エラー: 必要な列 ('{', '.join(missing)}') がデータフレームに存在しません。")
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        # --- データ準備 ---
        df_actual_calc = df_actual[[lead_time_col, actual_usage_col, price_col]].copy()
        df_pred_calc = df_predicted[[lead_time_col, pred_usage_col]].copy()

        # データ型を数値に変換（エラー時はNaN -> 0）
        for col in [actual_usage_col, price_col]:
            df_actual_calc[col] = pd.to_numeric(df_actual_calc[col], errors='coerce').fillna(0)
        df_pred_calc[pred_usage_col] = pd.to_numeric(df_pred_calc[pred_usage_col], errors='coerce').fillna(0)

        # リードタイムでマージして比較可能にする（外部マージで全リードタイムを保持）
        df_merged = pd.merge(df_actual_calc, df_pred_calc, on=lead_time_col, how='outer')
        df_merged = df_merged.sort_values(by=lead_time_col, ascending=False) # ★★★ 降順ソート ★★★

        # --- ★★★ Correct Fixed Price Determination ★★★ ---
        fixed_price = np.nan # Initialize
        # Find the price *before* the change (lead time > change_lead_time)
        data_before_change = df_merged[df_merged[lead_time_col] > change_lead_time]
        if not data_before_change.empty:
            # Price just before change is the first row in this descending sorted df
            fixed_price = data_before_change[price_col].iloc[0]
        else:
            # Fallback if no data strictly *before* change_lead_time
            # Use price *at* change_lead_time if available
            price_at_change_row = df_merged[df_merged[lead_time_col] == change_lead_time]
            if not price_at_change_row.empty:
                fixed_price = price_at_change_row[price_col].iloc[0]
                st.warning(f"売上予測: 変更前の価格が見つからず、変更時点(LT={change_lead_time})の価格 {fixed_price:.1f} を固定価格として使用します。", icon="⚠️")
            # Final fallback: use price at max lead time if still NaN
            elif not df_merged.empty:
                 fixed_price = df_merged[price_col].iloc[0]
                 st.warning(f"売上予測: 変更前/変更時の価格が見つからず、最大リードタイムの価格 {fixed_price:.1f} を固定価格として使用します。", icon="⚠️")
            else:
                 st.error("売上予測: 固定価格を決定できませんでした。")
                 return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Ensure fixed_price is a valid float, default to 0 if not
        if pd.isna(fixed_price):
            st.error(f"売上予測: 有効な固定価格を特定できませんでした (NaN)。予測売上は0になります。")
            fixed_price = 0.0
        else:
            fixed_price = float(fixed_price) # Ensure it's float

        st.info(f"予測売上計算に使用する固定価格: {fixed_price:.1f}") # Log the fixed price used
        # --- Fixed Price Determination End --- 

        # --- 日次新規予約数と日次売上の計算 (降順ループ) ---
        results = []
        prev_actual_usage = 0
        prev_pred_usage = 0
        total_actual_revenue_before_change = 0
        total_actual_revenue_after_change = 0
        total_predicted_revenue_after_change = 0
        for index, row in df_merged.iterrows():
            current_lt = row[lead_time_col]
            current_actual_usage = row[actual_usage_col]
            current_pred_usage = row[pred_usage_col]
            current_actual_price = row[price_col]

            # clip(lower=0) で利用台数減によるマイナス予約数を防ぐ
            actual_new = (current_actual_usage - prev_actual_usage)
            pred_new = (current_pred_usage - prev_pred_usage)

            actual_daily_revenue = actual_new * current_actual_price if actual_new > 0 else 0

            predicted_daily_revenue = 0
            if current_lt > change_lead_time: # Before price change
                predicted_daily_revenue = actual_daily_revenue
                total_actual_revenue_before_change += actual_daily_revenue
            elif current_lt <= change_lead_time: # At or after price change
                daily_pred_rev_after = pred_new * fixed_price if pred_new > 0 else 0
                predicted_daily_revenue = daily_pred_rev_after
                total_predicted_revenue_after_change += daily_pred_rev_after
                total_actual_revenue_after_change += actual_daily_revenue

            results.append({
                lead_time_col: current_lt,
                'actual_new_bookings': actual_new if actual_new > 0 else 0,
                'pred_new_bookings': pred_new if pred_new > 0 else 0,
                'actual_price': current_actual_price,
                'fixed_price': fixed_price, # Price used for prediction after change
                'actual_revenue': actual_daily_revenue,
                'predicted_revenue_calc': predicted_daily_revenue # Store the hybrid calculation
            })
            prev_actual_usage = current_actual_usage
            prev_pred_usage = current_pred_usage

        revenue_df = pd.DataFrame(results)
        # 必要に応じて change_lead_time 以前にフィルタリング (ただし全期間計算しても合計は同じはず)
        # revenue_df = revenue_df[revenue_df[lead_time_col] <= change_lead_time]

        # 累計と差額の計算
        revenue_df = revenue_df.sort_values(by=lead_time_col, ascending=True) # 累計計算のため昇順に戻す
        revenue_df['actual_revenue_cumsum'] = revenue_df['actual_revenue'].cumsum()
        # Rename the calculation column for clarity in the detailed df output
        revenue_df = revenue_df.rename(columns={'predicted_revenue_calc': 'predicted_revenue'})
        revenue_df['predicted_revenue_cumsum'] = revenue_df['predicted_revenue'].cumsum() # Cumsum of the hybrid value
        revenue_df['revenue_difference'] = revenue_df['actual_revenue'] - revenue_df['predicted_revenue']
        revenue_df['revenue_difference_cumsum'] = revenue_df['revenue_difference'].cumsum()
        revenue_df = revenue_df.sort_values(by=lead_time_col, ascending=False) # 表示用に降順に戻す

        # 合計値
        total_actual = total_actual_revenue_before_change + total_actual_revenue_after_change

        # ★★★ Apply logic: If actual sales increase after change is negligible, difference is 0 ★★★
        tolerance = 1e-9 # Define a small tolerance for floating point comparisons
        if total_actual_revenue_after_change <= tolerance: # Check against tolerance
             st.info(f"価格変更後 (LT <= {change_lead_time}) の実績売上増がほぼゼロのため、売上差額は0として扱います (実績売上後: {total_actual_revenue_after_change:.4f})", icon="ℹ️") # Log the value
             total_predicted_hybrid = total_actual # Predicted = Actual if no actual increase after change
             total_difference = 0
             total_predicted_revenue_after_change = 0 # Set predicted after to 0 as well
        else:
             # Original hybrid calculation if actual sales *did* increase after change
             total_predicted_hybrid = total_actual_revenue_before_change + total_predicted_revenue_after_change
             total_difference = total_actual - total_predicted_hybrid

        # ★★★ Return new values (rounded) ★★★
        return (revenue_df,
                round(total_actual, 0),
                round(total_predicted_hybrid, 0),
                round(total_difference, 0),
                round(total_actual_revenue_before_change, 0),
                round(total_actual_revenue_after_change, 0),
                round(total_predicted_revenue_after_change, 0))

    except Exception as e:
        st.error(f"売上計算中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


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