# 可視化関連の関数
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st # エラー表示用


def plot_booking_curve(df, x_col, y_col, title="予約曲線"):
    """指定された列を用いて予約曲線（折れ線グラフ）を描画する"""
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"エラー: グラフ描画に必要な列 ('{x_col}' または '{y_col}') がデータフレームに存在しません。")
        return go.Figure() # 空のFigureを返す
    try:
        fig = px.line(df, x=x_col, y=y_col, title=title,
                      labels={x_col: 'リードタイム（日）', y_col: '利用台数累積'})
        fig.update_layout(xaxis_title='利用日までの日数（0が当日）', yaxis_title='利用台数累積', xaxis_autorange='reversed')
        return fig
    except Exception as e:
        st.error(f"予約曲線の描画中にエラーが発生しました: {e}")
        return go.Figure()

def plot_price_trends(df, x_col, y_cols, title="価格推移"):
    """指定された列を用いて価格推移（複数の折れ線グラフ）を描画する"""
    missing_cols = [col for col in [x_col] + y_cols if col not in df.columns]
    if missing_cols:
        st.error(f"エラー: グラフ描画に必要な列 ('{', '.join(missing_cols)}') がデータフレームに存在しません。")
        return go.Figure()

    try:
        # Plotly Expressで複数の線を簡単に描画
        df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name='価格種別', value_name='価格')
        fig = px.line(df_melted, x=x_col, y='価格', color='価格種別', title=title,
                      labels={x_col: 'リードタイム（日）', '価格': '価格'})
        fig.update_layout(xaxis_title='利用日までの日数（0が当日）', yaxis_title='価格', xaxis_autorange='reversed')
        return fig
    except Exception as e:
        st.error(f"価格推移グラフの描画中にエラーが発生しました: {e}")
        return go.Figure()

def plot_comparison_curve(df_actual, df_predicted, x_col, y_actual_col, y_pred_col, title="実績 vs 予測"):
    """実績データと予測データを比較する折れ線グラフを描画する"""

    required_cols_actual = [x_col, y_actual_col]
    required_cols_predicted = [x_col, y_pred_col]

    missing_actual = [col for col in required_cols_actual if col not in df_actual.columns]
    missing_predicted = [col for col in required_cols_predicted if col not in df_predicted.columns]

    if missing_actual or missing_predicted:
        missing = list(set(missing_actual + missing_predicted))
        st.error(f"エラー: 比較グラフ描画に必要な列 ('{', '.join(missing)}') がデータフレームに存在しません。")
        return go.Figure()

    try:
        fig = go.Figure()

        # 実績データを追加
        fig.add_trace(go.Scatter(x=df_actual[x_col], y=df_actual[y_actual_col],
                                mode='lines', name='実績値',
                                line=dict(color='blue')))

        # 予測データを追加
        fig.add_trace(go.Scatter(x=df_predicted[x_col], y=df_predicted[y_pred_col],
                                mode='lines', name='予測値（価格変動なし）',
                                line=dict(color='red', dash='dash')))

        fig.update_layout(
            title=title,
            xaxis_title='利用日までの日数（0が当日）',
            yaxis_title='利用台数累積',
            legend_title="データ種別",
            xaxis_autorange='reversed'
        )

        return fig
    except Exception as e:
        st.error(f"比較グラフの描画中にエラーが発生しました: {e}")
        return go.Figure() 