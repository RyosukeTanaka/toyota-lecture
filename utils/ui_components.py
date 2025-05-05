# utils/ui_components.py
import streamlit as st
import pandas as pd
import datetime
from typing import Tuple, List, Dict, Any, Optional, Union # 型ヒント用

# --- 定数 (app.pyから一部移動/共有が必要な場合) ---
# これらはapp.pyで定義し、引数として渡す方が良いかもしれないが、
# ここでは簡単のため一部を定義
CAR_CLASS_COLUMN = '車両クラス'
DATE_COLUMN = '利用日'
TARGET_VARIABLE = '利用台数累積' # 予測用
BOOKING_DATE_COLUMN = '予約日' # カウント分析用
LEAD_TIME_COLUMN = 'リードタイム_計算済'

def render_prediction_sidebar_widgets(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Optional[str], Optional[datetime.date], List[str], List[str], List[str], List[str], bool]:
    """予測・比較分析ページのサイドバーウィジェットを描画し、選択値を返す"""
    selected_car_class: Optional[str] = "全クラス"
    selected_date: Optional[datetime.date] = None
    selected_numeric: List[str] = []
    selected_categorical: List[str] = []
    selected_features: List[str] = []
    models_to_compare: List[str] = []
    run_analysis: bool = False

    st.header("予測分析 設定")

    # --- 車両クラスの選択 ---
    if CAR_CLASS_COLUMN in data.columns:
        available_classes = data[CAR_CLASS_COLUMN].unique()
        class_options = ["全クラス"] + sorted(list(available_classes))
        selected_car_class = st.selectbox(
            f"'{CAR_CLASS_COLUMN}'を選択:",
            options=class_options, index=0, key="pred_class_select"
        )
    else:
        st.warning(f"'{CAR_CLASS_COLUMN}'列が見つかりません。")
        selected_car_class = "全クラス" # 値を確定

    # 選択された車両クラスでデータをフィルタリング (日付選択用)
    if selected_car_class == "全クラス":
        data_for_date_selection = data
    else:
        data_for_date_selection = data[data[CAR_CLASS_COLUMN] == selected_car_class]

    # --- 分析日の選択 ---
    if DATE_COLUMN in data_for_date_selection.columns and pd.api.types.is_datetime64_any_dtype(data_for_date_selection[DATE_COLUMN]):
        available_dates = data_for_date_selection[DATE_COLUMN].dt.date.unique()
        if len(available_dates) > 0:
            date_options_str = ['日付を選択'] + sorted([d.strftime('%Y-%m-%d') for d in available_dates])
            selected_date_str = st.selectbox(
                f"'{DATE_COLUMN}'を選択:",
                options=date_options_str, index=0, key="pred_date_select"
            )
            if selected_date_str != '日付を選択':
                try:
                    selected_date = pd.to_datetime(selected_date_str).date()
                except ValueError:
                    st.error("選択された日付の形式が無効です。")
                    selected_date = None # Noneにリセット
        # else: # データがない場合の情報はメインページで表示するためここでは省略
        #     st.info(f"'{selected_car_class}'クラスには有効な'{DATE_COLUMN}'データがありません。")
    else:
        st.warning(f"'{DATE_COLUMN}'列がないか日付型ではありません。")
        selected_date = None # Noneにリセット

    # --- モデル設定 (日付選択後) --- #
    if selected_date is not None:
        st.markdown("---")
        st.subheader("予測モデル設定")
        # 特徴量選択
        # 予測に直接使わない可能性のある列を除外
        potential_features = [col for col in data.columns if col not in [
            TARGET_VARIABLE, DATE_COLUMN, BOOKING_DATE_COLUMN, LEAD_TIME_COLUMN, 'リードタイム', '利用台数' # 利用台数も除外
        ]]
        # 利用可能な数値・カテゴリ特徴量を取得
        numeric_cols = data_for_date_selection[potential_features].select_dtypes(include=['number']).columns.tolist()
        category_cols = data_for_date_selection[potential_features].select_dtypes(exclude=['number', 'datetime', 'timedelta']).columns.tolist() # datetimeなども除外

        # 数値特徴量の選択 (config適用)
        numeric_defaults_config = config.get('default_numeric_features')
        if numeric_defaults_config is None:
            valid_default_numeric = numeric_cols
        elif isinstance(numeric_defaults_config, list):
            valid_default_numeric = [f for f in numeric_defaults_config if f in numeric_cols]
            if not valid_default_numeric and numeric_defaults_config: st.warning(f"Configの数値特徴量無効")
        else:
            st.error(f"Config数値特徴量設定エラー")
            valid_default_numeric = numeric_cols
        selected_numeric = st.multiselect("数値特徴量:", numeric_cols, default=valid_default_numeric, key="pred_num_feat")

        # カテゴリ特徴量の選択 (config適用)
        categorical_defaults_config = config.get('default_categorical_features', [])
        if not isinstance(categorical_defaults_config, list):
            st.error(f"Configカテゴリ特徴量設定エラー")
            categorical_defaults_config = []
        valid_default_categorical = [f for f in categorical_defaults_config if f in category_cols]
        if not valid_default_categorical and categorical_defaults_config: st.warning(f"Configカテゴリ特徴量無効")
        selected_categorical = st.multiselect("カテゴリ特徴量:", category_cols, default=valid_default_categorical, key="pred_cat_feat")

        selected_features = selected_numeric + selected_categorical

        # 評価モデル選択 (config適用)
        available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
        default_models_to_compare_from_config = config.get('default_models_to_compare')
        default_models = default_models_to_compare_from_config if default_models_to_compare_from_config else ['xgboost', 'lightgbm']
        valid_default_models = [m for m in default_models if m in available_models]
        models_to_compare = st.multiselect(
            "評価したいモデル:", available_models, default=valid_default_models, key="pred_models"
        )

        # 実行ボタン
        run_analysis = st.button("📊 予測・分析実行", key="run_pred_analysis")

    return selected_car_class, selected_date, selected_numeric, selected_categorical, selected_features, models_to_compare, run_analysis


def render_data_analysis_sidebar_widgets(data: pd.DataFrame) -> Tuple[Optional[datetime.date], bool]:
    """データ分析・修正ページのサイドバーウィジェットを描画し、選択日付とボタン状態を返す"""
    selected_analysis_date: Optional[datetime.date] = None
    analyze_button: bool = False

    st.header("データ分析 設定")

    # 予約日選択ウィジェット
    min_booking_date = None
    max_booking_date = None
    if BOOKING_DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[BOOKING_DATE_COLUMN]):
        valid_dates = data[BOOKING_DATE_COLUMN].dropna()
        if not valid_dates.empty:
            min_booking_date = valid_dates.min().date()
            max_booking_date = valid_dates.max().date()

    # 日付選択。範囲が取得できた場合のみ表示
    if min_booking_date and max_booking_date:
         # デフォルト日付を設定 (2025-04-15)
         default_date_val = datetime.date(2025, 4, 15)
         # デフォルトが範囲外の場合のフォールバック
         if not (min_booking_date <= default_date_val <= max_booking_date):
             st.warning(f"デフォルト日付 {default_date_val} がデータ範囲外 ({min_booking_date} ~ {max_booking_date}) のため、最も古い予約日を使用します。")
             default_date_val = min_booking_date

         selected_analysis_date = st.date_input(
             "起点となる予約日を選択してください:",
             value=default_date_val, # デフォルト日付を設定
             min_value=min_booking_date,
             max_value=max_booking_date,
             key="analysis_booking_date"
         )

         analyze_button = st.button("📈 分析実行", key="analyze_data_button")
    else:
        st.warning(f"分析に必要な '{BOOKING_DATE_COLUMN}' 列がないか、有効な日付データが含まれていません。")
        selected_analysis_date = None # Noneにリセット
        analyze_button = False

    return selected_analysis_date, analyze_button 