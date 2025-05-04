import streamlit as st
import pandas as pd
import os
import yaml # 追加
# PyCaretのインポートは modeling.py に移動するため、ここでは不要になる可能性があります
# from pycaret.regression import setup, compare_models, pull, save_model

# --- ユーティリティ関数のインポート ---
from utils.data_processing import (load_data,
                                   preprocess_data,
                                   display_exploration,
                                   filter_data_by_date,
                                   create_scenario_data,
                                   find_last_price_change_lead_time)
from utils.visualization import (plot_booking_curve,
                                 plot_price_trends,
                                 plot_comparison_curve,
                                 plot_feature_importance)
from utils.modeling import (setup_and_compare_models,
                          predict_with_model,
                          get_feature_importance_df)

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積' # 目的変数を定数化
DATE_COLUMN = '利用日'       # 分析の基準となる日付列
PRICE_COLUMNS = ['価格_トヨタ', '価格_オリックス'] # 価格比較対象の列
LEAD_TIME_COLUMN = 'リードタイム_計算済' # 前処理で計算したリードタイム列
CAR_CLASS_COLUMN = '車両クラス' # 車両クラス列を追加

# --- 設定ファイルを読み込む関数 --- #
# @st.cache_data # 設定ファイルの内容はキャッシュする -> 一時的に無効化
def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        st.success(f"設定ファイル '{config_path}' を読み込みました。")
        return config.get('model_config', {}) # model_config セクションを取得, なければ空辞書
    except FileNotFoundError:
        st.warning(f"設定ファイル '{config_path}' が見つかりません。デフォルト値を使用します。")
        return {}
    except Exception as e:
        st.error(f"設定ファイル '{config_path}' の読み込み中にエラーが発生しました: {e}")
        return {}
# ---------------------------- #

# --- Streamlit アプリケーション --- #
st.set_page_config(layout="wide") # ページレイアウトをワイドに設定
st.title("利用台数予測と比較分析システム")

# --- 設定読み込み --- #
config = load_config()
default_numeric_features_from_config = config.get('default_numeric_features')
default_categorical_features_from_config = config.get('default_categorical_features', []) # デフォルトは空リスト
default_models_to_compare_from_config = config.get('default_models_to_compare') # モデルリストも取得
# ----------------- #

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

            # --- モデル設定 (日付選択後) --- #
            if selected_date is not None:
                st.markdown("---")
                st.subheader("予測モデル設定")
                # 特徴量選択
                potential_features = [col for col in data.columns if col not in [TARGET_VARIABLE, DATE_COLUMN, '予約日', LEAD_TIME_COLUMN, 'リードタイム']]
                numeric_cols = data_for_date_selection[potential_features].select_dtypes(include=['number']).columns.tolist()
                category_cols = data_for_date_selection[potential_features].select_dtypes(exclude=['number']).columns.tolist()
                st.write("予測に使用する特徴量:")

                # --- 数値特徴量のデフォルト設定 --- #
                # configから読み込んだ値を取得 (キーが存在しない場合はNone)
                numeric_defaults_config = config.get('default_numeric_features')

                if numeric_defaults_config is None:
                    # configでnullまたはキー未定義の場合、利用可能な全てをデフォルトに
                    valid_default_numeric = numeric_cols
                    st.info("数値特徴量: config未指定のため全てデフォルト選択") # Debug
                elif isinstance(numeric_defaults_config, list):
                    # configでリスト指定の場合、利用可能なものだけをデフォルトに
                    valid_default_numeric = [f for f in numeric_defaults_config if f in numeric_cols]
                    if not valid_default_numeric and numeric_defaults_config: # configに指定はあるが有効なものがない場合
                         st.warning(f"config.yamlの数値特徴量 {numeric_defaults_config} は現在のデータに存在しません。")
                else:
                    # 想定外の型の場合 (エラーメッセージを出すなど)
                    st.error(f"config.yamlのdefault_numeric_featuresの値が無効です: {numeric_defaults_config}")
                    valid_default_numeric = numeric_cols # フォールバックとして全て選択

                # デバッグ用に渡すデフォルト値を表示
                st.write("Debug: Default numeric features passed to multiselect:", valid_default_numeric)
                selected_numeric = st.multiselect("数値特徴量:", numeric_cols, default=valid_default_numeric)
                # --------------------------------- #

                # --- カテゴリ特徴量のデフォルト設定 (同様にチェック強化) --- #
                categorical_defaults_config = config.get('default_categorical_features', []) # なければ空リスト
                if not isinstance(categorical_defaults_config, list):
                     st.error(f"config.yamlのdefault_categorical_featuresの値が無効です: {categorical_defaults_config}")
                     categorical_defaults_config = [] # 不正な値なら空にする

                valid_default_categorical = [f for f in categorical_defaults_config if f in category_cols]
                if not valid_default_categorical and categorical_defaults_config:
                     st.warning(f"config.yamlのカテゴリ特徴量 {categorical_defaults_config} は現在のデータに存在しません。")

                st.write("Debug: Default categorical features passed to multiselect:", valid_default_categorical) # デバッグ用
                selected_categorical = st.multiselect("カテゴリ特徴量:", category_cols, default=valid_default_categorical)
                # ----------------------------------- #

                selected_features = selected_numeric + selected_categorical

                # --- 評価モデル選択 (デフォルトをconfigから) --- #
                available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
                # configに指定があればそれ、なければ従来のデフォルト ['xgboost', 'lightgbm']
                default_models = default_models_to_compare_from_config if default_models_to_compare_from_config else ['xgboost', 'lightgbm']
                # 利用可能なモデルのみをデフォルトとする
                valid_default_models = [m for m in default_models if m in available_models]
                models_to_compare = st.multiselect(
                    "評価したいモデル:",
                    available_models,
                    default=valid_default_models # 修正: config値を反映
                )
                # ---------------------------------------- #

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

        # 再度データフィルタリング
        if selected_car_class == "全クラス":
            data_class_filtered_for_viz = data
        else:
            data_class_filtered_for_viz = data[data[CAR_CLASS_COLUMN] == selected_car_class]
        data_filtered = filter_data_by_date(data_class_filtered_for_viz, DATE_COLUMN, selected_date)

        if not data_filtered.empty:
            if LEAD_TIME_COLUMN in data_filtered.columns:
                data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)

                # --- 実際の予約曲線と価格推移 (変更なし) --- #
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
                if 'run_analysis' in locals() and run_analysis:
                    st.markdown("---")
                    st.header("モデル評価と予測比較")
                    with st.spinner('モデル比較と予測を実行中...'):
                        # --- 学習データ準備 ---
                        data_for_modeling = data[data[CAR_CLASS_COLUMN] == selected_car_class] if selected_car_class != "全クラス" else data

                        # --- 学習データサンプル表示 (修正) --- #
                        with st.expander("モデル学習に使用したデータのサンプルを表示 (選択された特徴量 + 目的変数)"):
                            # UIで選択された特徴量 + 目的変数 だけを抽出して表示
                            columns_to_show = selected_features + [TARGET_VARIABLE]
                            # data_for_modeling にこれらの列が存在するか確認してから表示
                            existing_columns_to_show = [col for col in columns_to_show if col in data_for_modeling.columns]
                            if existing_columns_to_show:
                                st.dataframe(data_for_modeling[existing_columns_to_show].head())
                            else:
                                st.warning("表示する列が見つかりません。特徴量を選択してください。")

                        # 1. モデル比較 (無視リストの生成ロジックを追加)
                        # 全ての利用可能な数値・カテゴリ特徴量を取得
                        all_numeric_cols = data_for_modeling[potential_features].select_dtypes(include=['number']).columns.tolist()
                        all_category_cols = data_for_modeling[potential_features].select_dtypes(exclude=['number']).columns.tolist()

                        # UIで選択 *されなかった* 特徴量を特定
                        ignored_numeric = list(set(all_numeric_cols) - set(selected_numeric))
                        ignored_categorical = list(set(all_category_cols) - set(selected_categorical))

                        # 以前エラーが出た列も無視リストに追加（重複はsetで解消される）
                        explicitly_ignored = ['曜日_name', 'en_name']
                        final_ignore_features = list(set(ignored_numeric + ignored_categorical + explicitly_ignored))

                        st.info(f"無視される特徴量: {final_ignore_features}") # デバッグ用に表示

                        # 利用する特徴量はUIで選択されたもののみ
                        valid_numeric = selected_numeric
                        valid_categorical = selected_categorical

                        best_model, comparison_results, setup_result = setup_and_compare_models(
                            _data=data_for_modeling,
                            target=TARGET_VARIABLE,
                            numeric_features=valid_numeric,       # UIで選択されたもの
                            categorical_features=valid_categorical, # UIで選択されたもの
                            ignore_features=final_ignore_features,  # 修正: 選択されなかったもの+α
                            include_models=models_to_compare,
                            sort_metric='RMSE'
                        )

                        if best_model is not None and setup_result is not None:
                            # --- 特徴量重要度の表示 --- #
                            st.markdown("---")
                            st.subheader(f"最良モデル ({type(best_model).__name__}) の特徴量重要度")
                            importance_df = get_feature_importance_df(best_model, setup_result)
                            if importance_df is not None and not importance_df.empty:
                                fig_importance = plot_feature_importance(importance_df)
                                if fig_importance:
                                     st.plotly_chart(fig_importance, use_container_width=True)
                                with st.expander("特徴量重要度データを表示"): 
                                     st.dataframe(importance_df)
                            else:
                                st.info("このモデルでは特徴量重要度データを取得/表示できません。")
                            # -------------------------

                            # --- 2. シナリオデータ作成 (最終価格固定シナリオ) ---
                            last_change_lt = find_last_price_change_lead_time(
                                data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN
                            )

                            predictions = pd.DataFrame() # 初期化
                            scenario_title_suffix = "" # タイトル用
                            data_scenario_created = False # シナリオデータが作られたか
                            data_scenario = pd.DataFrame() # 予測データサンプル表示用

                            if last_change_lt is not None:
                                st.write(f"価格最終変更リードタイム: {last_change_lt}")
                                data_scenario = create_scenario_data(
                                    data_filtered_sorted,
                                    PRICE_COLUMNS,
                                    LEAD_TIME_COLUMN,
                                    scenario_type='last_change_fixed',
                                    change_lead_time=last_change_lt
                                )
                                scenario_title_suffix = f"価格最終変更点(LT={last_change_lt})固定シナリオ"

                                # --- 予測データサンプル表示 --- #
                                if not data_scenario.empty:
                                    with st.expander(f"予測に使用したデータ ({scenario_title_suffix}) のサンプルを表示"):
                                        # 表示する列を選択された特徴量+必須列に限定
                                        display_columns = selected_features.copy()
                                        # リードタイム列、目的変数、価格列は必ず表示
                                        essential_columns = [LEAD_TIME_COLUMN, TARGET_VARIABLE] + PRICE_COLUMNS
                                        for col in essential_columns:
                                            if col in data_scenario.columns and col not in display_columns:
                                                display_columns.append(col)
                                        # 存在する列だけを抽出
                                        existing_display_columns = [col for col in display_columns if col in data_scenario.columns]
                                        if existing_display_columns:
                                            st.dataframe(data_scenario[existing_display_columns].head())
                                            
                                            # データダウンロード機能を追加
                                            download_data = data_scenario[existing_display_columns].copy()
                                            csv = download_data.to_csv(index=False)
                                            filename = f"scenario_data_{selected_date}_{selected_car_class}_{scenario_title_suffix.replace(' ', '_')}.csv"
                                            st.download_button(
                                                label="📊 予測用データをダウンロード (.csv)",
                                                data=csv,
                                                file_name=filename,
                                                mime="text/csv",
                                            )
                                        else:
                                            st.warning("表示する列が見つかりません。特徴量を選択してください。")
                                # -------------------------

                                # 3. 予測実行
                                if not data_scenario.empty:
                                     predictions = predict_with_model(best_model, data_scenario)
                                     if not predictions.empty:
                                         data_scenario_created = True
                                     else:
                                         st.error("予測の実行に失敗しました。")
                                else:
                                     st.error("最終価格固定シナリオデータの作成に失敗しました。")

                            else:
                                st.warning("価格変動が見つからなかったため、最終価格固定シナリオでの予測は実行できません。")

                            # 4. 結果表示
                            st.markdown("---")
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.subheader("モデル評価比較結果")
                                st.dataframe(comparison_results)
                            with col_m2:
                                if data_scenario_created and last_change_lt is not None: # 予測成功かつ最終変更点があった場合
                                    st.subheader(f"実績 vs 予測比較 ({scenario_title_suffix}) - LT {last_change_lt} 以降") # タイトル変更

                                    # --- 表示期間をフィルタリング --- #
                                    actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                    # 予測結果にもリードタイムが含まれているか確認し、フィルタリング
                                    if LEAD_TIME_COLUMN not in predictions.columns:
                                         predictions_with_lt = pd.merge(predictions, data_scenario[[LEAD_TIME_COLUMN]], left_index=True, right_index=True, how='left')
                                    else:
                                         predictions_with_lt = predictions
                                    predictions_filtered_display = predictions_with_lt[predictions_with_lt[LEAD_TIME_COLUMN] <= last_change_lt]
                                    # --------------------------------

                                    if not actual_filtered_display.empty and not predictions_filtered_display.empty:
                                        fig_compare = plot_comparison_curve(
                                            df_actual=actual_filtered_display, # フィルタ済みデータを使用
                                            df_predicted=predictions_filtered_display, # フィルタ済みデータを使用
                                            x_col=LEAD_TIME_COLUMN,
                                            y_actual_col=TARGET_VARIABLE,
                                            y_pred_col='prediction_label',
                                            title=f"{selected_date} {selected_car_class} 実績 vs 予測 ({scenario_title_suffix} - LT {last_change_lt} 以降)"
                                        )
                                        st.plotly_chart(fig_compare, use_container_width=True)

                                        # --- 実績 vs 予測データテーブル表示 (フィルタ済み) --- #
                                        st.subheader(f"実績 vs 予測 データテーブル ({scenario_title_suffix} - LT {last_change_lt} 以降)")
                                        df_actual_for_table = actual_filtered_display[[LEAD_TIME_COLUMN, TARGET_VARIABLE]].rename(
                                            columns={TARGET_VARIABLE: '実績利用台数'}
                                        )
                                        df_pred_for_table = predictions_filtered_display[[LEAD_TIME_COLUMN, 'prediction_label']].rename(
                                            columns={'prediction_label': '予測利用台数'}
                                        )
                                        df_comparison_table = pd.merge(
                                            df_actual_for_table,
                                            df_pred_for_table,
                                            on=LEAD_TIME_COLUMN,
                                            how='inner'
                                        )
                                        df_comparison_table = df_comparison_table.sort_values(by=LEAD_TIME_COLUMN)
                                        st.dataframe(df_comparison_table.reset_index(drop=True))
                                        # ----------------------------------------
                                    else:
                                         st.warning(f"リードタイム {last_change_lt} 以降の表示データがありません。")

                                elif last_change_lt is None:
                                     pass # 価格変動なし警告済み
                                else:
                                    st.error("予測結果を表示できませんでした。")
                        elif best_model is None:
                             st.error("モデル比較に失敗したため、予測を実行できませんでした。")
                        else: # setup_result is None (通常は比較失敗に含まれるはずだが念のため)
                             st.error("PyCaretのセットアップに失敗したため、処理を続行できません。")
            else:
                 st.warning(f"警告: リードタイム列 '{LEAD_TIME_COLUMN}' が見つかりません。グラフを表示できません。")
        else:
            st.info(f"選択された日付 '{selected_date}' ({selected_car_class}) のデータが見つかりませんでした。")
    elif uploaded_file is not None and selected_date is None and len(available_dates)>0:
         st.info("サイドバーから分析したい利用日を選択してください。")
    elif uploaded_file is not None and len(available_dates)==0:
         st.warning(f"アップロードされたファイルまたは選択された車両クラス '{selected_car_class}' には有効な利用日データが含まれていないようです。") 