# app.py 完成版 (データ更新機能・セッションステート対応)

import streamlit as st
import pandas as pd
import os
import yaml
import datetime
import numpy as np # データ更新用にインポート

# --- ユーティリティ関数のインポート ---
from utils.data_processing import (
    load_data, preprocess_data, display_exploration, filter_data_by_date,
    create_scenario_data, find_last_price_change_lead_time,
    recalculate_lag_feature # ラグ再計算関数
)
from utils.visualization import (
    plot_booking_curve, plot_price_trends, plot_comparison_curve,
    plot_feature_importance
)
from utils.modeling import (
    setup_and_compare_models, predict_with_model, get_feature_importance_df
)
from utils.ui_components import (
    render_prediction_sidebar_widgets, render_data_analysis_sidebar_widgets,
    render_model_training_sidebar_widgets # 新しく追加したウィジェット
)
from utils.analysis import analyze_unique_count_after_date, analyze_daily_sum_after_date
from utils.data_modification import nullify_usage_data_after_date # データ更新関数
# --- 新しいモジュールからのインポート ---
from utils.constants import (
    LAG_TARGET_COLUMN, LAG_DAYS, BOOKING_DATE_COLUMN, LAG_GROUP_COLS
)
from utils.page_prediction import render_prediction_analysis_page # 予測ページ関数
from utils.page_analysis import render_data_analysis_page     # 分析ページ関数
from utils.page_model_training import render_model_training_page # 新しく追加したモデルトレーニングページ関数
from utils.model_storage import ensure_model_directory # モデル保存ディレクトリ確保

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積'
DATE_COLUMN = '利用日'
PRICE_COLUMNS = ['価格_トヨタ', '価格_オリックス']
LEAD_TIME_COLUMN = 'リードタイム_計算済'
CAR_CLASS_COLUMN = '車両クラス'
USAGE_COUNT_COLUMN = '利用台数'
LAG_GROUP_COLS = [DATE_COLUMN, CAR_CLASS_COLUMN] # ラグ計算時のグループ化列

# --- 設定ファイルを読み込む関数 ---
def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('model_config', {})
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.sidebar.error(f"設定ファイル読み込みエラー: {e}")
        return {}

# --- アプリケーション本体 (main関数) ---
def main():
    st.set_page_config(layout="wide")
    config = load_config()
    
    # モデル保存ディレクトリを確保
    ensure_model_directory()

    # --- セッションステートの初期化 ---
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None
        st.session_state['last_uploaded_filename'] = None
    if 'zero_cutoff_date' not in st.session_state:
         st.session_state['zero_cutoff_date'] = None
    if 'last_update_summary' not in st.session_state:
        st.session_state['last_update_summary'] = None

    # --- サイドバー --- #
    with st.sidebar:
        # st.image("logo.png", width=100)
        st.title("分析メニュー")
        app_mode = st.radio(
            "実行したい分析を選択してください:",
            ("モデルトレーニング", "予測分析", "データ分析・修正"),
            key="app_mode_select"
        )
        st.markdown("---")
        st.header("データアップロード")
        uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

    # --- データ読み込み/処理 --- #
    if uploaded_file is not None:
        if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
             st.session_state['processed_data'] = None
             st.session_state['zero_cutoff_date'] = None
             st.session_state['last_update_summary'] = None
             st.session_state['last_uploaded_filename'] = uploaded_file.name
             st.info("新しいファイルがアップロードされたため、データを再処理します。")

        if st.session_state['processed_data'] is None:
            with st.spinner("データを読み込み・前処理中..."):
                data_raw = load_data(uploaded_file)
                if data_raw is not None:
                    data_processed_base = preprocess_data(data_raw) # preprocessの結果を一時変数に
                    if data_processed_base is not None and not data_processed_base.empty:
                         st.info("初期ラグ特徴量を計算中...")
                         # ★★★ recalculate_lag_feature の返り値をアンパック ★★★
                         data_processed_final, lag_info = recalculate_lag_feature(
                              df_processed=data_processed_base, # preprocess後のデータを渡す
                              lag_target_col=LAG_TARGET_COLUMN,
                              lag_days=LAG_DAYS,
                              booking_date_col=BOOKING_DATE_COLUMN,
                              group_cols=LAG_GROUP_COLS
                         )
                         # ★★★ セッションステートには DataFrame のみを保存 ★★★
                         if data_processed_final is not None:
                             st.session_state['processed_data'] = data_processed_final
                             st.sidebar.success("データ準備完了")
                         else:
                             # ラグ計算でエラーが起きた場合
                             st.error("ラグ特徴量の計算に失敗しました。データを確認してください。")
                             st.session_state['processed_data'] = None # エラー時はNone
                             st.stop()
                    else:
                         # preprocess_data が空のDFなどを返した場合
                         st.error("データの前処理に失敗しました。")
                         st.session_state['processed_data'] = None
                         st.stop()
                else:
                    st.error("データの読み込みに失敗しました。")
                    st.session_state['processed_data'] = None
                    st.stop()

    # --- ページ表示 --- #
    current_data = st.session_state.get('processed_data')
    if current_data is not None:
        if app_mode == "モデルトレーニング":
            render_model_training_page(current_data, config)  # 新しく追加したページ
        elif app_mode == "予測分析":
            render_prediction_analysis_page(current_data, config)  # 更新したページ
        elif app_mode == "データ分析・修正":
            render_data_analysis_page(current_data)           # 既存のままのページ
        else:
            st.error("無効なモードが選択されました。")
    elif uploaded_file is None:
         st.info("サイドバーから分析対象のCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()