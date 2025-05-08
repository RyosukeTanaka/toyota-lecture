# 可視化関連の関数
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st # エラー表示用

# --- 画像保存用のインポート ---
import os
import plotly.io as pio

# --- write_image が使えるようにするため ---
# デフォルトのレンダラーを設定（kaleido必須）
try:
    pio.renderers.default = "png"
except Exception as e:
    st.warning(f"画像保存機能の初期化に失敗しました: {e}")
    st.info("バッチ分析結果の画像保存には pip install kaleido コマンドを実行してください。")

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

def plot_feature_importance(df_importance):
    """特徴量重要度のDataFrameからPlotly棒グラフを作成する"""
    if df_importance is None or df_importance.empty:
        st.warning("特徴量重要度データが空のため、グラフを作成できません。")
        return None
    try:
        # 上位N件などに絞る場合はここでフィルタリング
        # df_importance = df_importance.head(20)

        fig = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h', # 横棒グラフ
            title='Feature Importance',
            # text='Importance' # バーに値を表示する場合
        )
        # 見やすいように並び順を逆転 (縦軸の上から重要度高)
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(df_importance) * 25) # 特徴量数に応じて高さを調整
        )
        # バーの値のフォーマット調整 (必要に応じて)
        # fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        return fig
    except Exception as e:
        st.error(f"特徴量重要度グラフの作成中にエラー: {e}")
        return None

def plot_full_period_comparison(df_actual, df_predicted, x_col, y_actual_col, y_pred_col, title=None, change_lead_time=None):
    """リードタイム全期間での実績vs予測比較グラフを生成"""
    required_cols_actual = [x_col, y_actual_col]
    required_cols_predicted = [x_col, y_pred_col]

    missing_actual = [col for col in required_cols_actual if col not in df_actual.columns]
    missing_predicted = [col for col in required_cols_predicted if col not in df_predicted.columns]

    if missing_actual or missing_predicted:
        missing = list(set(missing_actual + missing_predicted))
        st.error(f"エラー: 全期間比較グラフ描画に必要な列 ('{', '.join(missing)}') がデータフレームに存在しません。")
        return go.Figure()
    
    try:
        fig = go.Figure()
        
        # 実績データプロット
        fig.add_trace(go.Scatter(
            x=df_actual[x_col],
            y=df_actual[y_actual_col],
            mode='lines+markers',
            name='実績値',
            line=dict(color='rgba(0, 123, 255, 0.8)', width=2),
            marker=dict(size=6)
        ))
        
        # 予測データプロット
        fig.add_trace(go.Scatter(
            x=df_predicted[x_col],
            y=df_predicted[y_pred_col],
            mode='lines+markers',
            name='予測値（価格固定シナリオ）',
            line=dict(color='rgba(220, 53, 69, 0.8)', width=2),
            marker=dict(size=6)
        ))
        
        # 価格変更点の表示
        if change_lead_time is not None:
            fig.add_vline(
                x=change_lead_time, 
                line_dash="dash", 
                line_color="rgba(40, 167, 69, 0.5)",
                annotation_text=f"価格最終変更点 (LT={change_lead_time})",
                annotation_position="bottom right"
            )
        
        # グラフレイアウト設定
        fig.update_layout(
            title=title or "全期間での実績 vs 予測比較",
            xaxis_title='利用日までの日数（0が当日）',
            yaxis_title='利用台数累積',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="x unified",
            xaxis_autorange='reversed'
        )
        
        return fig
    except Exception as e:
        st.error(f"全期間比較グラフの描画中にエラーが発生しました: {e}")
        return go.Figure()

def plot_batch_revenue_comparison(
    df: pd.DataFrame, 
    x_col: str,
    horizontal: bool = False,
    title: str = "売上比較分析"
) -> go.Figure:
    """バッチ処理結果の売上比較グラフを描画する

    Parameters
    ----------
    df : pd.DataFrame
        集計済みの売上データフレーム（実績売上、予測売上、売上差額の列が必要）
    x_col : str
        x軸に使用する列名（利用日や車両クラスなど）
    horizontal : bool, default=False
        横向きのグラフにするかどうか
    title : str, default="売上比較分析"
        グラフのタイトル

    Returns
    -------
    go.Figure
        Plotlyのグラフオブジェクト
    """
    if df.empty or x_col not in df.columns:
        st.error(f"グラフ描画に必要なデータが不足しています。")
        return go.Figure()
    
    try:
        # 差額の絶対値の最大値を計算
        max_diff = df["売上差額"].abs().max()
        
        # 値のフォーマット関数
        def format_value(val):
            return f"{int(val):,}円"
            
        if horizontal:
            # 横向きのバーチャート（x/y軸を入れ替え）
            fig = go.Figure()
            
            # 実績売上バー
            fig.add_trace(go.Bar(
                y=df[x_col],
                x=df["実績売上"],
                name="実績売上",
                orientation="h",
                marker_color="rgba(55, 128, 191, 0.7)",
                text=[format_value(val) for val in df["実績売上"]],
                textposition="auto"
            ))
            
            # 予測売上バー
            fig.add_trace(go.Bar(
                y=df[x_col],
                x=df["予測売上"],
                name="予測売上（価格固定）",
                orientation="h",
                marker_color="rgba(219, 64, 82, 0.7)",
                text=[format_value(val) for val in df["予測売上"]],
                textposition="auto"
            ))
            
            # 差額マーカー
            fig.add_trace(go.Scatter(
                y=df[x_col],
                x=df["売上差額"],
                name="売上差額",
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color=["green" if v >= 0 else "red" for v in df["売上差額"]],
                    line=dict(width=2, color="DarkSlateGrey")
                ),
                text=[format_value(val) for val in df["売上差額"]],
                textposition="middle right"
            ))
            
            # レイアウト
            fig.update_layout(
                title=title,
                xaxis_title="売上金額",
                yaxis_title=x_col,
                barmode="group",
                height=max(400, 100 + 50*len(df)),  # 項目数に応じて高さを調整
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=60, b=20)
            )
        else:
            # 縦向きのバーチャート
            fig = go.Figure()
            
            # 実績売上バー
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=df["実績売上"],
                name="実績売上",
                marker_color="rgba(55, 128, 191, 0.7)",
                text=[format_value(val) for val in df["実績売上"]],
                textposition="auto"
            ))
            
            # 予測売上バー
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=df["予測売上"],
                name="予測売上（価格固定）",
                marker_color="rgba(219, 64, 82, 0.7)",
                text=[format_value(val) for val in df["予測売上"]],
                textposition="auto"
            ))
            
            # 差額マーカー
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df["売上差額"],
                name="売上差額",
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color=["green" if v >= 0 else "red" for v in df["売上差額"]],
                    line=dict(width=2, color="DarkSlateGrey")
                ),
                text=[format_value(val) for val in df["売上差額"]],
                textposition="top center"
            ))
            
            # レイアウト
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title="売上金額",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=60, b=20)
            )
        
        # X軸日付なら適切な表示形式を設定
        if x_col == "利用日" and not horizontal:
            fig.update_xaxes(type="date", tickformat="%Y-%m-%d")
        
        # ファイル保存用に設定を追加
        fig.update_layout(width=1200, height=800)  # 保存用に大きなサイズを設定
        
        return fig
        
    except Exception as e:
        st.error(f"バッチ結果グラフの描画中にエラー: {e}")
        return go.Figure()

def plot_price_change_lead_time_distribution(df_price_changes: pd.DataFrame, group_by_col: str, lead_time_col: str = "変更リードタイム", title_prefix: str = "") -> go.Figure:
    """価格変更リードタイムの分布を箱ひげ図で表示する"""
    if df_price_changes.empty or group_by_col not in df_price_changes.columns or lead_time_col not in df_price_changes.columns:
        st.warning(f"価格変更リードタイム分布グラフ: データまたは必要な列 ('{group_by_col}', '{lead_time_col}') がありません。")
        return go.Figure()
    
    try:
        fig = px.box(
            df_price_changes,
            x=group_by_col,
            y=lead_time_col,
            color=group_by_col,
            title=f"{title_prefix}{group_by_col}別 価格変更リードタイムの分布",
            labels={lead_time_col: "変更リードタイム (日)", group_by_col: group_by_col},
            points="all" # 全てのデータポイントも表示
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
        )
        if pd.api.types.is_datetime64_any_dtype(df_price_changes[group_by_col]):
            fig.update_xaxes(title_text="利用日") # X軸が日付の場合のラベル修正
        return fig
    except Exception as e:
        st.error(f"価格変更リードタイム分布グラフの作成中にエラー: {e}")
        return go.Figure()

def plot_price_change_magnitude_scatter(df_price_changes: pd.DataFrame, lead_time_col: str = "変更リードタイム", car_class_col: str = "車両クラス", title: str = "リードタイムと価格変動幅の関係") -> go.Figure:
    """変更リードタイムと価格変動幅の関係を散布図で表示する"""
    required_cols = [lead_time_col, "変更前価格", "変更後価格", car_class_col]
    if df_price_changes.empty or not all(col in df_price_changes.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_price_changes.columns]
        st.warning(f"価格変動幅散布図: データまたは必要な列 ('{', '.join(missing)}') がありません。")
        return go.Figure()
    
    try:
        df_plot = df_price_changes.copy()
        df_plot["価格変動幅"] = df_plot["変更後価格"] - df_plot["変更前価格"]
        df_plot["価格変動幅_Abs"] = df_plot["価格変動幅"].abs()
        
        fig = px.scatter(
            df_plot,
            x=lead_time_col,
            y="価格変動幅",
            color=car_class_col,
            size="価格変動幅_Abs",
            hover_data=["利用日", "変更前価格", "変更後価格"],
            title=title,
            labels={lead_time_col: "変更リードタイム (日)", "価格変動幅": "価格変動幅 (円)"}
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_autorange='reversed' # リードタイムなので逆順
        )
        return fig
    except Exception as e:
        st.error(f"価格変動幅散布図の作成中にエラー: {e}")
        return go.Figure()

def plot_price_change_vs_booking_impact(df: pd.DataFrame, x_col: str = "価格変更幅", y_col: str = "追加予約数インパクト", color_col: str = "車両クラス", title="価格変更幅 vs 追加予約数インパクト") -> go.Figure:
    """価格変更幅と追加予約数インパクトの関係を散布図で表示する"""
    required_cols = ["変更前価格", "変更後価格", "追加実績予約数（価格変更後）", "追加予測予約数（価格が変更されなかった場合）", color_col, "利用日"]
    if df.empty or not all(col in df.columns for col in required_cols):
        st.warning(f"価格変更インパクトグラフ: データまたは必要な列 {required_cols} が不足しています。")
        return go.Figure()

    df_plot = df.copy()
    # Calculate derived columns, handling potential errors
    try:
        df_plot[x_col] = pd.to_numeric(df_plot["変更後価格"], errors='coerce') - pd.to_numeric(df_plot["変更前価格"], errors='coerce')
        df_plot[y_col] = pd.to_numeric(df_plot["追加実績予約数（価格変更後）"], errors='coerce') - pd.to_numeric(df_plot["追加予測予約数（価格が変更されなかった場合）"], errors='coerce')
    except Exception as e:
        st.error(f"価格変更インパクト計算中にエラー: {e}")
        return go.Figure()

    # Drop rows where calculation resulted in NaN (due to missing inputs)
    df_plot = df_plot.dropna(subset=[x_col, y_col])

    if df_plot.empty:
        st.info("価格変更幅と追加予約数の両方が有効なデータ点がありません。")
        return go.Figure()

    try:
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=["利用日", "追加実績予約数（価格変更後）", "追加予測予約数（価格が変更されなかった場合）", "変更前価格", "変更後価格"],
            title=title,
            labels={x_col: "価格変更幅 (円)", y_col: "追加予約数インパクト (実績 - 予測)"}
        )

        # Add a horizontal line at y=0 for reference
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        # Add a vertical line at x=0 for reference
        fig.add_vline(x=0, line_dash="dash", line_color="grey")

        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig
    except Exception as e:
        st.error(f"価格変更インパクトグラフ描画中にエラー: {e}")
        return go.Figure() 