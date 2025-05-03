import streamlit as st
import pandas as pd
import os
# PyCaretのインポートは modeling.py に移動するため、ここでは不要になる可能性があります
# from pycaret.regression import setup, compare_models, pull, save_model

# --- ユーティリティ関数のインポート ---
from utils.data_processing import (load_data,
                                   preprocess_data,
                                   display_exploration,
                                   filter_data_by_date,
                                   create_scenario_data)
from utils.visualization import (plot_booking_curve,
                                 plot_price_trends,
                                 plot_comparison_curve)
from utils.modeling import (setup_and_compare_models,
                          predict_with_model)

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積' # 目的変数を定数化
DATE_COLUMN = '利用日'       # 分析の基準となる日付列
PRICE_COLUMNS = ['価格_トヨタ', '価格_オリックス'] # 価格比較対象の列
LEAD_TIME_COLUMN = 'リードタイム_計算済' # 前処理で計算したリードタイム列
CAR_CLASS_COLUMN = '車両クラス' # 車両クラス列を追加

# --- Streamlit アプリケーション --- #
st.set_page_config(layout="wide") # ページレイアウトをワイドに設定
st.title("利用台数予測と比較分析システム")

# --- サイドバー --- #
with st.sidebar:
    st.header("設定")
    st.write("予測・分析対象のCSVファイルをアップロードしてください:")
    uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

    # ファイルがアップロードされたら、データ処理と残りの設定項目を表示
    data = None
    data_raw = None
    selected_car_class = "全クラス"
    selected_date = None
    available_dates = []
    data_for_date_selection = pd.DataFrame() # 空で初期化
    numeric_cols = []
    category_cols = []
    potential_features = []

    if uploaded_file is not None:
        data_raw = load_data(uploaded_file)
        if data_raw is not None:
            data = preprocess_data(data_raw)

            st.markdown("---")
            # --- 車両クラスの選択 ---
            if CAR_CLASS_COLUMN in data.columns:
                available_classes = data[CAR_CLASS_COLUMN].unique()
                class_options = ["全クラス"] + sorted(list(available_classes))
                selected_car_class = st.selectbox(
                    f"分析したい'{CAR_CLASS_COLUMN}'を選択してください:",
                    options=class_options,
                    index=0
                )
            else:
                st.warning(f"'{CAR_CLASS_COLUMN}'列が見つかりません。")
                selected_car_class = "全クラス"

            # 選択された車両クラスでデータをフィルタリング
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
                        f"分析したい'{DATE_COLUMN}'を選択してください:",
                        options=date_options_str,
                        index=0
                    )
                    if selected_date_str == '日付を選択':
                        selected_date = None
                    else:
                        try:
                            selected_date = pd.to_datetime(selected_date_str).date()
                        except ValueError:
                            st.error("選択された日付の形式が無効です。")
                            selected_date = None
                else:
                    st.info(f"'{selected_car_class}'クラスには有効な'{DATE_COLUMN}'データがありません。")
                    selected_date = None
            else:
                st.warning(f"'{DATE_COLUMN}'列がないか日付型ではありません。")
                selected_date = None

            # --- モデル設定 (日付選択後) ---
            if selected_date is not None:
                st.markdown("---")
                st.subheader("予測モデル設定")
                # 特徴量選択
                potential_features = [col for col in data.columns if col not in [TARGET_VARIABLE, DATE_COLUMN, '予約日', LEAD_TIME_COLUMN, 'リードタイム']]
                numeric_cols = data_for_date_selection[potential_features].select_dtypes(include=['number']).columns.tolist()
                category_cols = data_for_date_selection[potential_features].select_dtypes(exclude=['number']).columns.tolist()
                st.write("予測に使用する特徴量:")
                selected_numeric = st.multiselect("数値特徴量:", numeric_cols, default=numeric_cols)
                selected_categorical = st.multiselect("カテゴリ特徴量:", category_cols, default=category_cols)
                selected_features = selected_numeric + selected_categorical

                # 評価モデル選択
                available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
                models_to_compare = st.multiselect(
                    "評価したいモデル:",
                    available_models,
                    default=['lr', 'rf', 'lightgbm']
                )
                # 実行ボタン
                run_analysis = st.button("分析・予測実行")

# --- メインエリア --- #
if uploaded_file is None:
    st.info("サイドバーからCSVファイルをアップロードしてください。")
elif data is None:
    st.error("データの読み込みまたは前処理に失敗しました。")
else:
    st.subheader("処理済みデータ（先頭5行）")
    st.dataframe(data.head())

    st.markdown("---")

    # --- データ探索 --- #
    display_exploration(data) # これはメインエリアでも良いか

    st.markdown("---")

    # --- 分析・予測結果表示 --- #
    if selected_date is not None:
        st.header(f"分析結果: {selected_date} ({selected_car_class})")

        # 再度データフィルタリング（キャッシュや状態管理を使う方が効率的だが、ここではシンプルに再実行）
        if selected_car_class == "全クラス":
            data_filtered = filter_data_by_date(data, DATE_COLUMN, selected_date)
        else:
            data_class_filtered = data[data[CAR_CLASS_COLUMN] == selected_car_class]
            data_filtered = filter_data_by_date(data_class_filtered, DATE_COLUMN, selected_date)

        if not data_filtered.empty:
            if LEAD_TIME_COLUMN in data_filtered.columns:
                data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)

                # --- 実際の予約曲線と価格推移 --- #
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("実際の予約曲線")
                    fig_actual = plot_booking_curve(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_col=TARGET_VARIABLE, title=f"{selected_date} {selected_car_class} 実際の予約曲線")
                    st.plotly_chart(fig_actual, use_container_width=True)
                with col2:
                    st.subheader("価格推移")
                    fig_prices = plot_price_trends(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_cols=PRICE_COLUMNS, title=f"{selected_date} {selected_car_class} 価格推移")
                    st.plotly_chart(fig_prices, use_container_width=True)

                # --- モデル比較と予測実行結果 --- #
                # run_analysis がサイドバーのボタンの状態を持つ
                if 'run_analysis' in locals() and run_analysis: # ボタンが押されたか確認
                    st.markdown("---")
                    st.header("モデル評価と予測比較")
                    with st.spinner('モデル比較と予測を実行中...'):
                        # 1. モデル比較
                        data_for_modeling = data[data[CAR_CLASS_COLUMN] == selected_car_class] if selected_car_class != "全クラス" else data
                        best_model, comparison_results = setup_and_compare_models(
                            _data=data_for_modeling,
                            target=TARGET_VARIABLE,
                            numeric_features=selected_numeric,
                            categorical_features=selected_categorical,
                            include_models=models_to_compare,
                            sort_metric='RMSE'
                        )

                        if best_model is not None:
                            # 2. シナリオデータ作成
                            data_scenario = create_scenario_data(data_filtered_sorted, PRICE_COLUMNS, scenario_type='mean')
                            # 3. 予測実行
                            predictions = predict_with_model(best_model, data_scenario)

                            # 4. 結果表示
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.subheader("モデル評価比較結果")
                                st.dataframe(comparison_results)
                            with col_m2:
                                if not predictions.empty:
                                    st.subheader("実績 vs 予測比較")
                                    fig_compare = plot_comparison_curve(
                                        df_actual=data_filtered_sorted,
                                        df_predicted=predictions,
                                        x_col=LEAD_TIME_COLUMN,
                                        y_actual_col=TARGET_VARIABLE,
                                        y_pred_col='prediction_label',
                                        title=f"{selected_date} {selected_car_class} 実績 vs 予測（価格変動なしシナリオ）"
                                    )
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                else:
                                    st.error("予測データの作成または予測の実行に失敗しました。")
                        else:
                            st.error("モデル比較に失敗したため、予測を実行できませんでした。")
            else:
                 st.warning(f"警告: リードタイム列 '{LEAD_TIME_COLUMN}' が見つかりません。グラフを表示できません。")
        else:
            st.info(f"選択された日付 '{selected_date}' ({selected_car_class}) のデータが見つかりませんでした。")
    elif uploaded_file is not None and selected_date is None and len(available_dates)>0:
         st.info("サイドバーから分析したい利用日を選択してください。")
    elif uploaded_file is not None and len(available_dates)==0:
         st.warning(f"アップロードされたファイルまたは選択された車両クラス '{selected_car_class}' には有効な利用日データが含まれていないようです。")

# --- ここから下の古いコードは削除またはコメントアウト ---
# st.title("利用台数予測と比較分析システム")

# st.write("予測・分析対象のCSVファイルをアップロードしてください:")

# uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

# if uploaded_file is not None:
#     # --- 1. データの読み込みと前処理 ---
#     data_raw = load_data(uploaded_file)

#     if data_raw is not None:
#         data = preprocess_data(data_raw)

#         st.write("処理済みデータ（最初の20行）:")
#         st.dataframe(data.head(20))

#         st.markdown("---")

#         # --- 2. データ探索 (utilsに関数を移動) ---
#         display_exploration(data)

#         st.markdown("---")

#         # --- 3a. 車両クラスの選択 ---
#         st.header("分析設定")
#         selected_car_class = "全クラス" # デフォルト
#         if CAR_CLASS_COLUMN in data.columns:
#             available_classes = data[CAR_CLASS_COLUMN].unique()
#             # オプションに「全クラス」を追加
#             class_options = ["全クラス"] + sorted(list(available_classes))
#             selected_car_class = st.selectbox(
#                 f"分析したい'{CAR_CLASS_COLUMN}'を選択してください:",
#                 options=class_options,
#                 index=0 # デフォルトで「全クラス」を選択
#             )
#         else:
#             st.warning(f"警告: '{CAR_CLASS_COLUMN}'列が見つかりません。車両クラスによる絞り込みはできません。")

#         # --- 3b. 分析日の選択 ---
#         # 選択された車両クラスでデータをフィルタリング（日付選択の前に）
#         if selected_car_class == "全クラス":
#             data_for_date_selection = data
#         else:
#             data_for_date_selection = data[data[CAR_CLASS_COLUMN] == selected_car_class]
#             st.info(f"'{selected_car_class}' クラスのデータに絞り込みました。")

#         selected_date = None # 初期化
#         if DATE_COLUMN in data_for_date_selection.columns and pd.api.types.is_datetime64_any_dtype(data_for_date_selection[DATE_COLUMN]):
#             # 利用可能な日付は絞り込んだデータから取得
#             available_dates = data_for_date_selection[DATE_COLUMN].dt.date.unique()
#             if len(available_dates) > 0:
#                 date_options_str = ['日付を選択'] + sorted([d.strftime('%Y-%m-%d') for d in available_dates])
#                 selected_date_str = st.selectbox(
#                     f"分析したい'{DATE_COLUMN}'を選択してください:",
#                     options=date_options_str,
#                     index=0 # デフォルトで「日付を選択」を選択
#                 )

#                 # 選択された文字列を日付オブジェクトまたはNoneに変換
#                 if selected_date_str == '日付を選択':
#                     selected_date = None
#                 else:
#                     try:
#                         selected_date = pd.to_datetime(selected_date_str).date()
#                     except ValueError:
#                         st.error("選択された日付の形式が無効です。")
#                         selected_date = None
#             else:
#                 st.warning(f"'{selected_car_class}' クラスには有効な'{DATE_COLUMN}'が見つかりません。")
#                 selected_date = None

#         else:
#             st.warning(f"警告: '{DATE_COLUMN}'列が見つからないか、日付型ではありません。日付選択は利用できません。")
#             selected_date = None

#         # --- 4. 特定日の分析と可視化 ---
#         if selected_date is not None:
#             st.subheader(f"{selected_date} の分析 ({selected_car_class})") # タイトルにクラス名を追加
#             # 選択された日付でデータをフィルタリング (車両クラスで絞り込んだデータから更に絞り込む)
#             data_filtered = filter_data_by_date(data_for_date_selection, DATE_COLUMN, selected_date)

#             if not data_filtered.empty:
#                 # リードタイムでソート
#                 if LEAD_TIME_COLUMN in data_filtered.columns:
#                     data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)

#                     # --- 実際の予約曲線と価格推移の可視化 ---
#                     st.write("実際の予約曲線と価格推移:")
#                     fig_actual = plot_booking_curve(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_col=TARGET_VARIABLE, title=f"{selected_date} {selected_car_class} 実際の予約曲線")
#                     st.plotly_chart(fig_actual, use_container_width=True)
#                     fig_prices = plot_price_trends(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_cols=PRICE_COLUMNS, title=f"{selected_date} {selected_car_class} 価格推移")
#                     st.plotly_chart(fig_prices, use_container_width=True)

#                 else:
#                     st.warning(f"警告: リードタイム列 '{LEAD_TIME_COLUMN}' が見つかりません。グラフを表示できません。")

#                 # --- 5. モデル構築と予測の設定 (UI部分のみ残す) ---
#                 st.markdown("---")
#                 st.header("予測モデルの設定と実行")

#                 # 利用可能な列を取得 (ターゲット変数、日付関連、リードタイム列を除外)
#                 potential_features = [col for col in data.columns if col not in [TARGET_VARIABLE, DATE_COLUMN, '予約日', LEAD_TIME_COLUMN, 'リードタイム']]
#                 # 数値とカテゴリに分ける（PyCaretの自動判別補助）
#                 # 絞り込んだデータから特徴量候補の列を取得する
#                 numeric_cols = data_for_date_selection[potential_features].select_dtypes(include=['number']).columns.tolist()
#                 category_cols = data_for_date_selection[potential_features].select_dtypes(exclude=['number']).columns.tolist()

#                 st.write("予測に使用する特徴量を選択してください:")
#                 selected_numeric = st.multiselect("数値特徴量:", numeric_cols, default=numeric_cols)
#                 selected_categorical = st.multiselect("カテゴリ特徴量:", category_cols, default=category_cols)
#                 selected_features = selected_numeric + selected_categorical

#                 available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
#                 models_to_compare = st.multiselect(
#                     "評価したいモデルを選択してください:",
#                     available_models,
#                     default=['lr', 'rf', 'lightgbm']
#                 )

#                 if st.button("モデル比較と予測実行"):
#                     if not selected_features:
#                         st.warning("特徴量を1つ以上選択してください。")
#                     elif not models_to_compare:
#                         st.warning("評価するモデルを1つ以上選択してください。")
#                     else:
#                          with st.spinner('モデル比較と予測を実行中...'):
#                             # 1. モデル比較
#                             # 車両クラスが選択されている場合は、そのクラスのデータのみでモデル比較を行う
#                             data_for_modeling = data_for_date_selection # 車両クラスで絞り込み済みのデータ
#                             best_model, comparison_results = setup_and_compare_models(
#                                 _data=data_for_modeling, # 修正: 車両クラスで絞り込んだデータを使用
#                                 target=TARGET_VARIABLE,
#                                 numeric_features=selected_numeric,
#                                 categorical_features=selected_categorical,
#                                 include_models=models_to_compare,
#                                 sort_metric='RMSE' # 例: RMSEでソート
#                             )

#                             if best_model is not None:
#                                 # 2. シナリオデータ作成 (選択された日のデータで)
#                                 # ここでは例として平均価格を使用するシナリオ
#                                 data_scenario = create_scenario_data(data_filtered_sorted, PRICE_COLUMNS, scenario_type='mean')

#                                 # 3. 予測実行
#                                 predictions = predict_with_model(best_model, data_scenario)

#                                 # 4. 結果表示
#                                 st.subheader("モデル評価比較結果")
#                                 st.dataframe(comparison_results)

#                                 if not predictions.empty:
#                                     st.subheader("予測結果との比較")
#                                     fig_compare = plot_comparison_curve(
#                                         df_actual=data_filtered_sorted,
#                                         df_predicted=predictions,
#                                         x_col=LEAD_TIME_COLUMN,
#                                         y_actual_col=TARGET_VARIABLE,
#                                         y_pred_col='prediction_label', # predict_modelのデフォルト出力列名
#                                         title=f"{selected_date} {selected_car_class} 実績 vs 予測（価格変動なしシナリオ）"
#                                     )
#                                     st.plotly_chart(fig_compare, use_container_width=True)
#                                 else:
#                                     st.error("予測データの作成または予測の実行に失敗しました。")
#                             else:
#                                 st.error("モデル比較に失敗したため、予測を実行できませんでした。")
#             else:
#                 st.info(f"{selected_date} のデータが見つかりませんでした。")
#         else:
#             st.info("サイドバーから分析したい利用日を選択してください。")
# else:
#     st.info('CSVファイルをアップロードすると、データが表示され、分析と予測ができます。') 