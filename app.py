import streamlit as st
import pandas as pd
import os
import yaml
import datetime

# --- ユーティリティ関数のインポート ---
from utils.data_processing import (
    load_data, preprocess_data, display_exploration, filter_data_by_date,
    create_scenario_data, find_last_price_change_lead_time
)
from utils.visualization import (
    plot_booking_curve, plot_price_trends, plot_comparison_curve,
    plot_feature_importance
)
from utils.modeling import (
    setup_and_compare_models, predict_with_model, get_feature_importance_df
)
from utils.ui_components import (
    render_prediction_sidebar_widgets, render_data_analysis_sidebar_widgets
)
from utils.analysis import analyze_unique_count_after_date, analyze_daily_sum_after_date

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積'
DATE_COLUMN = '利用日'
PRICE_COLUMNS = ['価格_トヨタ', '価格_オリックス']
LEAD_TIME_COLUMN = 'リードタイム_計算済'
CAR_CLASS_COLUMN = '車両クラス'
BOOKING_DATE_COLUMN = '予約日'
USAGE_COUNT_COLUMN = '利用台数'

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

# --- 予測・比較分析ページ描画関数 ---
def render_prediction_analysis_page(data: pd.DataFrame, config: dict):
    st.title("予測・比較分析")

    # --- サイドバーウィジェットの描画と値の取得 ---
    (
        selected_car_class,
        selected_date,
        selected_numeric,
        selected_categorical,
        selected_features,
        models_to_compare,
        run_analysis
    ) = render_prediction_sidebar_widgets(data, config)

    # --- メインエリア --- #
    st.subheader("処理済みデータプレビュー")
    st.dataframe(data.head())
    st.markdown("---")

    if selected_date is not None:
        st.header(f"分析結果: {selected_date} ({selected_car_class})")

        # データフィルタリング
        if selected_car_class == "全クラス":
            data_class_filtered_for_viz = data
        else:
            data_class_filtered_for_viz = data[data[CAR_CLASS_COLUMN] == selected_car_class]

        # 利用日でのフィルタリング
        data_filtered = filter_data_by_date(data_class_filtered_for_viz, DATE_COLUMN, selected_date)

        if not data_filtered.empty:
            if LEAD_TIME_COLUMN in data_filtered.columns:
                data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)

                # 実際の予約曲線と価格推移グラフ
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("実際の予約曲線")
                    fig_actual = plot_booking_curve(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_col=TARGET_VARIABLE, title=f"{selected_date} {selected_car_class} 実際の予約曲線")
                    st.plotly_chart(fig_actual, use_container_width=True)
                with col2:
                    st.subheader("価格推移")
                    fig_prices = plot_price_trends(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_cols=PRICE_COLUMNS, title=f"{selected_date} {selected_car_class} 価格推移")
                    st.plotly_chart(fig_prices, use_container_width=True)

                # モデル比較と予測の実行 (ボタンが押された場合)
                if run_analysis:
                    st.markdown("---")
                    st.header("モデル評価と予測比較")
                    with st.spinner('モデル比較と予測を実行中...'):
                        # 学習用データの準備 (車両クラスでフィルタ)
                        data_for_modeling = data[data[CAR_CLASS_COLUMN] == selected_car_class] if selected_car_class != "全クラス" else data

                        # 学習データサンプルの表示
                        with st.expander("モデル学習に使用したデータのサンプルを表示"):
                            columns_to_show = selected_features + [TARGET_VARIABLE]
                            existing_columns_to_show = [col for col in columns_to_show if col in data_for_modeling.columns]
                            if existing_columns_to_show:
                                st.dataframe(data_for_modeling[existing_columns_to_show].head())
                            else:
                                st.warning("表示する列が見つかりません。特徴量を選択してください。")

                        # 無視する特徴量のリストを作成
                        potential_features = [col for col in data.columns if col not in [
                            TARGET_VARIABLE, DATE_COLUMN, BOOKING_DATE_COLUMN, LEAD_TIME_COLUMN, 'リードタイム', '利用台数'
                        ]]
                        all_numeric_cols = data_for_modeling[potential_features].select_dtypes(include=['number']).columns.tolist()
                        all_category_cols = data_for_modeling[potential_features].select_dtypes(exclude=['number', 'datetime', 'timedelta']).columns.tolist()
                        ignored_numeric = list(set(all_numeric_cols) - set(selected_numeric))
                        ignored_categorical = list(set(all_category_cols) - set(selected_categorical))
                        explicitly_ignored = ['曜日_name', 'en_name', '利用台数累積_lag30'] # lag30_recalcを使う場合は元も無視
                        final_ignore_features = list(set(ignored_numeric + ignored_categorical + explicitly_ignored))

                        # モデル比較の実行
                        best_model, comparison_results, setup_result = setup_and_compare_models(
                            _data=data_for_modeling, target=TARGET_VARIABLE,
                            numeric_features=selected_numeric, categorical_features=selected_categorical,
                            ignore_features=final_ignore_features, include_models=models_to_compare,
                            sort_metric='RMSE'
                        )

                        if best_model is not None and setup_result is not None:
                            # 特徴量重要度の表示
                            st.markdown("---")
                            st.subheader(f"最良モデル ({type(best_model).__name__}) の特徴量重要度")
                            importance_df = get_feature_importance_df(best_model, setup_result)
                            if importance_df is not None and not importance_df.empty:
                                fig_importance = plot_feature_importance(importance_df)
                                if fig_importance: st.plotly_chart(fig_importance, use_container_width=True)
                                with st.expander("特徴量重要度データ"): st.dataframe(importance_df)
                            else:
                                st.info("このモデルでは特徴量重要度を表示できません。")

                            # 詳細CV結果表示 (utils/modeling.py内で処理される)
                            # modeling.py内の st.subheader("クロスバリデーション詳細結果...") で表示

                            # シナリオデータ作成と予測
                            last_change_lt = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)
                            predictions = pd.DataFrame()
                            scenario_title_suffix = ""
                            data_scenario_created = False
                            data_scenario = pd.DataFrame()

                            if last_change_lt is not None:
                                st.write(f"価格最終変更リードタイム: {last_change_lt}")
                                data_scenario = create_scenario_data(
                                    data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN,
                                    scenario_type='last_change_fixed', change_lead_time=last_change_lt
                                )
                                scenario_title_suffix = f"価格最終変更点(LT={last_change_lt})固定シナリオ"

                                if not data_scenario.empty:
                                    with st.expander(f"予測に使用したデータ ({scenario_title_suffix}) のサンプル"):
                                        display_columns = selected_features + [LEAD_TIME_COLUMN, TARGET_VARIABLE] + PRICE_COLUMNS
                                        existing_display_columns = [col for col in display_columns if col in data_scenario.columns]
                                        if existing_display_columns:
                                            st.dataframe(data_scenario[existing_display_columns].head())
                                            csv = data_scenario[existing_display_columns].to_csv(index=False)
                                            filename = f"scenario_data_{selected_date}_{selected_car_class}_{scenario_title_suffix.replace(' ', '_')}.csv"
                                            st.download_button("💾 予測用データをダウンロード", csv, filename, "text/csv")
                                        else: st.warning("表示列なし")

                                    predictions = predict_with_model(best_model, data_scenario)
                                    if not predictions.empty: data_scenario_created = True
                                    else: st.error("予測実行失敗")
                                else: st.error("シナリオデータ作成失敗")
                            else:
                                st.warning("価格変動が見つからなかったため、最終価格固定シナリオでの予測は実行できません。")

                            # 結果表示
                            st.markdown("---")
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.subheader("モデル評価比較結果")
                                st.dataframe(comparison_results) # pull()の結果を表示
                            with col_m2:
                                if data_scenario_created and last_change_lt is not None:
                                    st.subheader(f"実績 vs 予測比較 ({scenario_title_suffix}) - LT {last_change_lt} 以降")
                                    # 表示期間をフィルタリング
                                    actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                    if LEAD_TIME_COLUMN not in predictions.columns:
                                        predictions_with_lt = pd.merge(predictions, data_scenario[[LEAD_TIME_COLUMN]], left_index=True, right_index=True, how='left')
                                    else: predictions_with_lt = predictions
                                    predictions_filtered_display = predictions_with_lt[predictions_with_lt[LEAD_TIME_COLUMN] <= last_change_lt]

                                    if not actual_filtered_display.empty and not predictions_filtered_display.empty:
                                        fig_compare = plot_comparison_curve(
                                            df_actual=actual_filtered_display, df_predicted=predictions_filtered_display,
                                            x_col=LEAD_TIME_COLUMN, y_actual_col=TARGET_VARIABLE, y_pred_col='prediction_label',
                                            title=f"{selected_date} {selected_car_class} 実績 vs 予測 (LT {last_change_lt} 以降)"
                                        )
                                        st.plotly_chart(fig_compare, use_container_width=True)

                                        st.subheader(f"実績 vs 予測 データテーブル (LT {last_change_lt} 以降)")
                                        df_actual_for_table = actual_filtered_display[[LEAD_TIME_COLUMN, TARGET_VARIABLE]].rename(columns={TARGET_VARIABLE: '実績利用台数'})
                                        df_pred_for_table = predictions_filtered_display[[LEAD_TIME_COLUMN, 'prediction_label']].rename(columns={'prediction_label': '予測利用台数'})
                                        df_comparison_table = pd.merge(df_actual_for_table, df_pred_for_table, on=LEAD_TIME_COLUMN, how='inner')
                                        st.dataframe(df_comparison_table.sort_values(by=LEAD_TIME_COLUMN).reset_index(drop=True))
                                    else:
                                        st.warning(f"リードタイム {last_change_lt} 以降の表示データがありません。")
                                elif last_change_lt is None:
                                     pass # 価格変動なしは上で警告済み
                                else:
                                    st.error("予測結果を表示できませんでした。")
                        elif best_model is None:
                             st.error("モデル比較に失敗したため、予測を実行できませんでした。")
                        else: # setup_result is None
                             st.error("PyCaretのセットアップに失敗したため、処理を続行できません。")
            else:
                 st.warning(f"警告: リードタイム列 '{LEAD_TIME_COLUMN}' が見つかりません。グラフを表示できません。")
        else:
            # 利用日が選択されているが、その日のデータが見つからない場合
            st.info(f"選択された日付 '{selected_date}' ({selected_car_class}) のデータが見つかりませんでした。")
    elif selected_date is None:
        # データがあり、利用日列も存在するが、日付が選択されていない場合にメッセージを表示
        # (available_dates のチェックはサイドバーで行われているため、ここでは selected_date is None だけで判断)
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]) and not data[DATE_COLUMN].isnull().all():
             st.info("サイドバーから分析したい利用日を選択してください。")
        # else: # 利用日列がない、または有効な日付がない場合はサイドバーで警告済み

# --- データ分析・修正ページ描画関数 ---
def render_data_analysis_page(data: pd.DataFrame):
    st.title("データ分析・修正")

    # --- サイドバーウィジェットの描画と値の取得 ---
    selected_analysis_date, analyze_button = render_data_analysis_sidebar_widgets(data)

    # --- メインエリア --- #
    st.header("データ探索")
    display_exploration(data)
    st.markdown("---")

    # --- 利用台数の日別合計グラフ表示 --- #
    st.header(f"特定予約日以降の '{USAGE_COUNT_COLUMN}' 日別合計推移")
    st.write(f"指定した日付より**後**の予約日について、日ごとの '{USAGE_COUNT_COLUMN}' の合計値をグラフ表示します。")

    if analyze_button and selected_analysis_date is not None:
        with st.spinner("分析中..."):
            # 日別合計を計算
            daily_sum_df = analyze_daily_sum_after_date(
                data=data,
                start_date=selected_analysis_date,
                booking_date_col=BOOKING_DATE_COLUMN,
                sum_col=USAGE_COUNT_COLUMN
            )

            if daily_sum_df is not None:
                if not daily_sum_df.empty:
                    st.success("分析完了！")
                    # 折れ線グラフで表示
                    st.line_chart(daily_sum_df)

                    # ★★★ 最初に0になった日付を特定して表示 ★★★
                    zero_date = None
                    # DataFrameを日付でソート（インデックスが日付のはず）
                    daily_sum_df_sorted = daily_sum_df.sort_index()
                    # 合計値の列名を取得 (例: '利用台数_合計')
                    sum_col_name = f'{USAGE_COUNT_COLUMN}_合計'
                    if sum_col_name in daily_sum_df_sorted.columns:
                        # 0以下の最初の行を探す (浮動小数点誤差を考慮する場合は <= 1e-9 など)
                        zero_rows = daily_sum_df_sorted[daily_sum_df_sorted[sum_col_name] <= 0]
                        if not zero_rows.empty:
                            zero_date = zero_rows.index[0].strftime('%Y-%m-%d') # 最初の日付を取得
                            st.info(f"📈グラフの値が最初に0になった（または下回った）日付: **{zero_date}**")
                        else:
                            st.info("📈グラフ期間中に値が0になる日はありませんでした。")
                    else:
                         st.warning("グラフデータから合計列が見つかりませんでした。")
                    # ---------------------------------------

                    # (オプション) データテーブルも表示
                    with st.expander("詳細データ表示"):
                         st.dataframe(daily_sum_df)
                # else: # データがない場合は analysis 関数内で info 表示
                #     pass
            # else: # analysis 関数内でエラー表示
            #    pass
    elif selected_analysis_date is None:
         st.info("サイドバーで起点となる予約日を確認してください。")

# --- アプリケーション本体 ---
def main():
    st.set_page_config(layout="wide")
    # 設定読み込み
    config = load_config()

    # --- サイドバー --- #
    with st.sidebar:
        # st.image("logo.png", width=100) # ロゴ表示 (コメントアウト)
        st.title("分析メニュー")
        app_mode = st.radio(
            "実行したい分析を選択してください:",
            ("予測・比較分析", "データ分析・修正"),
            key="app_mode_select"
        )
        st.markdown("---")
        st.header("データアップロード")
        uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

    # --- データ読み込みと前処理 --- #
    data = None
    if uploaded_file is not None:
        # データ読み込み・前処理は選択モードに関わらず最初に行う
        with st.spinner("データを読み込み・前処理中..."):
            data_raw = load_data(uploaded_file)
            if data_raw is not None:
                data = preprocess_data(data_raw)
            else:
                st.error("データの読み込みに失敗しました。")
                st.stop()
    else:
        st.info("サイドバーから分析対象のCSVファイルをアップロードしてください。")
        st.stop()

    # --- 選択されたモードに応じてページを表示 --- #
    if data is not None:
        if app_mode == "予測・比較分析":
            render_prediction_analysis_page(data, config)
        elif app_mode == "データ分析・修正":
            render_data_analysis_page(data)
        else:
            st.error("無効なモードが選択されました。")

if __name__ == "__main__":
    main() 