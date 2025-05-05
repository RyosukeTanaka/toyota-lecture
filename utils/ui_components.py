# utils/ui_components.py
import streamlit as st
import pandas as pd
import datetime
from typing import Tuple, List, Dict, Any, Optional, Union # 型ヒント用
from .model_storage import list_saved_models # モデル一覧取得用

# --- 定数 (app.pyから一部移動/共有が必要な場合) ---
# これらはapp.pyで定義し、引数として渡す方が良いかもしれないが、
# ここでは簡単のため一部を定義
CAR_CLASS_COLUMN = '車両クラス'
DATE_COLUMN = '利用日'
TARGET_VARIABLE = '利用台数累積' # 予測用
BOOKING_DATE_COLUMN = '予約日' # カウント分析用
LEAD_TIME_COLUMN = 'リードタイム_計算済'

def render_prediction_sidebar_widgets(data: pd.DataFrame) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
    """予測ページのサイドバーウィジェットを描画し、選択値を返す"""
    selected_car_class: Optional[str] = "全クラス"
    selected_model_info: Optional[Dict[str, Any]] = None
    run_prediction: bool = False

    st.header("予測分析 設定")

    # --- 保存済みモデルの選択 ---
    saved_models = list_saved_models()
    if not saved_models:
        st.warning("保存済みモデルが見つかりません。「モデルトレーニング」ページでモデルを作成してください。")
        return selected_car_class, None, False
    
    model_names = [model["model_name"] for model in saved_models]
    selected_model_name = st.selectbox(
        "使用するモデルを選択:",
        options=model_names,
        index=0 if model_names else None,
        key="pred_model_select"
    )
    
    # 選択されたモデル情報を取得
    selected_model_info = next((model for model in saved_models if model["model_name"] == selected_model_name), None)
    
    # 選択したモデル情報の表示（モデルが選択された時点で表示）
    if selected_model_info:
        st.markdown("---")
        st.subheader("選択したモデル情報")
        
        # モデル基本情報
        st.info(f"モデル名: {selected_model_info['model_name']}")
        st.info(f"モデルタイプ: {selected_model_info['model_type']}")
        st.info(f"作成日: {selected_model_info['creation_date']}")
        
        # モデル性能指標（存在する場合）
        if "metrics" in selected_model_info:
            st.subheader("モデル性能指標")
            metrics = selected_model_info["metrics"]
            metrics_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value
                with metrics_cols[i]:
                    st.metric(metric_name, formatted_value)
    
    # 選択されたモデルの車両クラスに基づいて、車両クラス選択肢をフィルタリング
    if selected_model_info:
        model_car_class = selected_model_info.get("car_class")
        
        # モデルが特定の車両クラス用の場合、その車両クラスのみを選択可能に
        if model_car_class != "全クラス":
            selected_car_class = model_car_class
            st.info(f"選択したモデルは '{model_car_class}' 専用のモデルです。")
        else:
            # モデルが全クラス対応の場合
            if CAR_CLASS_COLUMN in data.columns:
                available_classes = data[CAR_CLASS_COLUMN].unique()
                class_options = ["全クラス"] + sorted(list(available_classes))
                selected_car_class = st.selectbox(
                    f"'{CAR_CLASS_COLUMN}'を選択:",
                    options=class_options, 
                    index=0, 
                    key="pred_class_select"
                )
            else:
                st.warning(f"'{CAR_CLASS_COLUMN}'列が見つかりません。")
                selected_car_class = "全クラス"

    # 予測実行ボタンはメインエリアでの日付選択後に表示するため、ここでは返さない
    return selected_car_class, selected_model_info, run_prediction


def render_model_training_sidebar_widgets(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[str, List[str], List[str], List[str], List[str], str, bool]:
    """モデルトレーニングページのサイドバーウィジェットを描画し、選択値を返す"""
    selected_car_class: str = "全クラス"
    selected_numeric: List[str] = []
    selected_categorical: List[str] = []
    selected_features: List[str] = []
    models_to_compare: List[str] = []
    model_name: str = ""
    run_training: bool = False

    st.header("モデルトレーニング 設定")

    # --- 車両クラスの選択 ---
    if CAR_CLASS_COLUMN in data.columns:
        available_classes = data[CAR_CLASS_COLUMN].unique()
        class_options = ["全クラス"] + sorted(list(available_classes))
        selected_car_class = st.selectbox(
            f"'{CAR_CLASS_COLUMN}'を選択:",
            options=class_options, index=0, key="train_class_select"
        )
    else:
        st.warning(f"'{CAR_CLASS_COLUMN}'列が見つかりません。")
        selected_car_class = "全クラス" # 値を確定

    # --- モデル設定 --- #
    st.markdown("---")
    st.subheader("モデル設定")
    
    # モデル名入力
    model_name = st.text_input("保存するモデル名:", key="model_name_input", placeholder="例: XGBoost_車両クラスA_20230401")
    
    # 予測に直接使わない可能性のある列を除外
    potential_features = [col for col in data.columns if col not in [
        TARGET_VARIABLE, DATE_COLUMN, BOOKING_DATE_COLUMN, LEAD_TIME_COLUMN, 'リードタイム', '利用台数' # 利用台数も除外
    ]]
    
    # 利用可能な数値・カテゴリ特徴量を取得
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in potential_features]
    
    category_cols = data.select_dtypes(exclude=['number', 'datetime', 'timedelta']).columns.tolist()
    category_cols = [col for col in category_cols if col in potential_features]

    # 数値特徴量の選択 (config適用)
    numeric_defaults_config = config.get('default_numeric_features')
    if numeric_defaults_config is None:
        valid_default_numeric = numeric_cols
    elif isinstance(numeric_defaults_config, list):
        valid_default_numeric = [f for f in numeric_defaults_config if f in numeric_cols]
        if not valid_default_numeric and numeric_defaults_config: 
            st.warning(f"Configの数値特徴量無効")
    else:
        st.error(f"Config数値特徴量設定エラー")
        valid_default_numeric = numeric_cols
    
    selected_numeric = st.multiselect("数値特徴量:", numeric_cols, default=valid_default_numeric, key="train_num_feat")

    # カテゴリ特徴量の選択 (config適用)
    categorical_defaults_config = config.get('default_categorical_features', [])
    if not isinstance(categorical_defaults_config, list):
        st.error(f"Configカテゴリ特徴量設定エラー")
        categorical_defaults_config = []
    
    valid_default_categorical = [f for f in categorical_defaults_config if f in category_cols]
    if not valid_default_categorical and categorical_defaults_config: 
        st.warning(f"Configカテゴリ特徴量無効")
    
    selected_categorical = st.multiselect("カテゴリ特徴量:", category_cols, default=valid_default_categorical, key="train_cat_feat")

    # 選択された特徴量を統合
    selected_features = selected_numeric + selected_categorical

    # 評価モデル選択 (config適用)
    available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
    default_models_to_compare_from_config = config.get('default_models_to_compare')
    default_models = default_models_to_compare_from_config if default_models_to_compare_from_config else ['xgboost', 'lightgbm']
    valid_default_models = [m for m in default_models if m in available_models]
    
    models_to_compare = st.multiselect(
        "評価したいモデル:", available_models, default=valid_default_models, key="train_models"
    )

    # 実行ボタン
    run_training = st.button("🧠 モデルトレーニング実行", key="run_training")

    return selected_car_class, selected_numeric, selected_categorical, selected_features, models_to_compare, model_name, run_training


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
         # デフォルト日付を設定 (2025-01-01 に変更)
         default_date_val = datetime.date(2025, 1, 1)
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