# utils/page_batch_analysis.py

import streamlit as st
import pandas as pd
import datetime
from typing import Dict, Any, List
from .constants import (
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN
)
from .ui_components import render_prediction_sidebar_widgets
from .model_storage import load_model, get_model_metadata
from .batch_analysis import run_batch_prediction, display_batch_results


def render_batch_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    """複数日付・車両クラスのバッチ分析ページを描画

    Parameters
    ----------
    data : pd.DataFrame
        分析用データフレーム
    config : Dict[str, Any]
        設定情報辞書
    """
    st.title("複数日付範囲での集計分析")
    
    # サイドバーウィジェットの描画と値の取得
    (
        selected_car_class,
        selected_model_info,
        _  # 予測実行ボタンはメインエリアで表示するため無視
    ) = render_prediction_sidebar_widgets(data)
    
    # --- メインエリア --- #
    if not selected_model_info:
        st.warning("予測を実行するには、まず「モデルトレーニング」ページでモデルを作成し、サイドバーで使用するモデルを選択してください。")
        return
    
    # --- バッチ分析の入力パラメータ設定 --- #
    st.header("分析対象設定")
    st.markdown("複数の日付と車両クラスの組み合わせに対して一括で予測分析を行います。")
    
    # 分析手法の説明を追加
    with st.expander("分析手法の説明", expanded=True):
        st.markdown("""
        ### 売上金額影響分析の仕組み
        
        この分析ツールは、**価格変更が売上に与えた影響**を定量的に評価します。計算方法は以下の通りです：
        
        1. **価格変更点の特定**：
           - 各日付・車両クラスごとに価格変更があった最後のリードタイム（LT）を特定
           - このポイントを基準点として分析
        
        2. **実績売上の計算**：
           - 実際に発生した予約データに基づいた売上
           - 各リードタイムでの日次新規予約数 × その時点の実際価格
        
        3. **予測売上の計算**：
           - 「価格が変更されなかった場合」のシナリオに基づく予測売上
           - 機械学習モデルが予測した日次新規予約数 × 価格変更点の固定価格
        
        4. **売上差額の算出**：
           - 実績売上 - 予測売上 = 価格変更による影響額
           - プラスの値：価格戦略により売上増加
           - マイナスの値：価格戦略により売上減少
        
        この分析により、価格変更の効果を数値で確認できます。例えば価格を上げたことで予約数が減っても総売上が増加したケースや、価格を下げて予約数を増やした効果を確認できます。
        """)
    
    # 日付範囲選択
    date_range = []
    with st.expander("日付範囲の選択", expanded=True):
        # 有効な日付リストを取得
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]):
            available_dates = sorted(data[DATE_COLUMN].dt.date.unique())
            
            if len(available_dates) > 0:
                # 日付選択方法
                date_selection_method = st.radio(
                    "日付選択方法",
                    ["日付範囲を指定", "個別の日付を選択"],
                    horizontal=True
                )
                
                if date_selection_method == "日付範囲を指定":
                    col1, col2 = st.columns(2)
                    with col1:
                        min_date = available_dates[0]
                        max_date = available_dates[-1]
                        start_date = st.date_input("開始日", min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("終了日", max_date, min_value=start_date, max_value=max_date)
                    
                    # 開始日と終了日の間の日付をすべて取得
                    date_range = [d for d in available_dates if start_date <= d <= end_date]
                    st.info(f"選択された日付範囲に含まれる利用日: {len(date_range)}日")
                    
                else:  # 個別の日付を選択
                    date_options = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in available_dates]
                    selected_dates = st.multiselect(
                        "分析する日付を選択",
                        options=date_options,
                        default=date_options[:min(5, len(date_options))]  # デフォルトは最初の5つ
                    )
                    date_range = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in selected_dates]
                    st.info(f"選択された日付: {len(date_range)}日")
                
                # 選択された日付を表示
                if date_range:
                    st.subheader("選択された日付一覧")
                    date_df = pd.DataFrame([{'利用日': d} for d in date_range])
                    st.dataframe(date_df)
                else:
                    st.warning("少なくとも1つの日付を選択してください")
            else:
                st.warning("利用可能な日付がありません")
        else:
            st.error(f"'{DATE_COLUMN}'列がないか日付型ではありません")
    
    # 車両クラス選択
    car_classes = []
    with st.expander("車両クラスの選択", expanded=True):
        if CAR_CLASS_COLUMN in data.columns:
            available_classes = sorted(data[CAR_CLASS_COLUMN].unique())
            
            # 車両クラス選択方法
            class_selection_method = st.radio(
                "車両クラス選択方法",
                ["すべての車両クラス", "個別の車両クラスを選択"],
                horizontal=True
            )
            
            if class_selection_method == "すべての車両クラス":
                car_classes = available_classes
                st.info(f"すべての車両クラス: {len(car_classes)}クラス")
            else:  # 個別の車両クラスを選択
                selected_classes = st.multiselect(
                    "分析する車両クラスを選択",
                    options=available_classes,
                    default=[available_classes[0]] if available_classes else []
                )
                car_classes = selected_classes
                st.info(f"選択された車両クラス: {len(car_classes)}クラス")
            
            # 選択された車両クラスを表示
            if car_classes:
                st.subheader("選択された車両クラス一覧")
                class_df = pd.DataFrame([{'車両クラス': c} for c in car_classes])
                st.dataframe(class_df)
            else:
                st.warning("少なくとも1つの車両クラスを選択してください")
        else:
            st.error(f"'{CAR_CLASS_COLUMN}'列が存在しません")
    
    # 実行設定
    with st.expander("実行設定", expanded=True):
        st.markdown("#### 並列処理の設定")
        max_workers = st.slider("並列処理数", min_value=1, max_value=8, value=4, 
                              help="同時に処理するタスクの数を指定します。大きすぎると処理が不安定になる可能性があります。")
        
        total_combinations = len(date_range) * len(car_classes)
        estimated_time = total_combinations * 1.5 / max_workers  # 1.5秒 × 組み合わせ数 / 並列数
        
        st.info(f"処理予定の組み合わせ: 日付 {len(date_range)}個 × 車両クラス {len(car_classes)}個 = 合計 {total_combinations}件")
        st.warning(f"処理時間目安: 約 {estimated_time:.1f}秒（{estimated_time/60:.1f}分）")
    
    # 実行ボタン
    run_batch = False
    if date_range and car_classes:
        run_batch = st.button("🚀 バッチ分析を実行", key="run_batch", 
                             help="選択された日付・車両クラスの組み合わせに対してバッチ処理を実行します")
    
    # バッチ処理を実行
    if run_batch:
        st.markdown("---")
        st.header("バッチ分析結果")
        
        # 選択された日付期間を表示
        if date_range:
            min_date = min(date_range)
            max_date = max(date_range)
            if min_date == max_date:
                st.subheader(f"分析期間: {min_date}")
            else:
                st.subheader(f"分析期間: {min_date} 〜 {max_date}")
        
        with st.spinner('バッチ処理を実行中...'):
            # モデルのロード
            model = load_model(selected_model_info["path"])
            
            if model is None:
                st.error("モデルの読み込みに失敗しました。")
                return
            
            # モデルのメタデータを取得
            model_metadata = None
            if "filename" in selected_model_info:
                model_metadata = get_model_metadata(selected_model_info["filename"])
            
            # バッチ予測実行
            predictions, metadata_list = run_batch_prediction(
                data=data,
                model=model,
                date_range=date_range,
                car_classes=car_classes,
                model_metadata=model_metadata,
                max_workers=max_workers
            )
            
            # 結果表示
            display_batch_results(metadata_list)
    
    # データプレビュー（常に表示しておく）
    st.markdown("---")
    st.subheader("データプレビュー")
    st.dataframe(data.head()) 