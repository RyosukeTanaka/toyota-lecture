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
    render_prediction_sidebar_widgets, render_data_analysis_sidebar_widgets
)
from utils.analysis import analyze_unique_count_after_date, analyze_daily_sum_after_date
from utils.data_modification import nullify_usage_data_after_date # データ更新関数

# --- 定数 ---
TARGET_VARIABLE = '利用台数累積'
DATE_COLUMN = '利用日'
PRICE_COLUMNS = ['価格_トヨタ', '価格_オリックス']
LEAD_TIME_COLUMN = 'リードタイム_計算済'
CAR_CLASS_COLUMN = '車両クラス'
BOOKING_DATE_COLUMN = '予約日'
USAGE_COUNT_COLUMN = '利用台数'
LAG_TARGET_COLUMN = '利用台数累積' # ラグ計算の対象列
LAG_DAYS = 30 # ラグ日数
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
    st.subheader("処理済みデータプレビュー (現在の状態)")
    st.dataframe(data.head())
    # データ探索は別ページ

    st.markdown("---") # プレビューと結果の間に区切り

    if selected_date is not None:
        st.header(f"分析結果: {selected_date} ({selected_car_class})")

        # データフィルタリング
        if selected_car_class == "全クラス":
            data_class_filtered_for_viz = data
        else:
            data_class_filtered_for_viz = data[data[CAR_CLASS_COLUMN] == selected_car_class]

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

                # モデル比較と予測の実行
                if run_analysis:
                    st.markdown("---")
                    st.header("モデル評価と予測比較")
                    with st.spinner('モデル比較と予測を実行中...'):
                        data_for_modeling = data[data[CAR_CLASS_COLUMN] == selected_car_class] if selected_car_class != "全クラス" else data

                        with st.expander("モデル学習に使用したデータのサンプルを表示"):
                            columns_to_show = selected_features.copy()
                            if TARGET_VARIABLE not in columns_to_show:
                                columns_to_show.append(TARGET_VARIABLE)

                            lag_col_name = f"{LAG_TARGET_COLUMN}_lag{LAG_DAYS}"
                            if lag_col_name in data_for_modeling.columns and lag_col_name not in columns_to_show:
                                columns_to_show.append(lag_col_name)

                            existing_columns_to_show = [col for col in columns_to_show if col in data_for_modeling.columns]
                            if pd.Series(existing_columns_to_show).duplicated().any():
                                existing_columns_to_show = pd.Series(existing_columns_to_show).drop_duplicates().tolist()

                            if existing_columns_to_show:
                                st.dataframe(data_for_modeling[existing_columns_to_show].head())
                            else:
                                st.warning("表示する列が見つかりません。")

                        # 無視リスト生成 (potential_features の再定義もここで行う)
                        potential_features = [col for col in data.columns if col not in [
                            TARGET_VARIABLE, DATE_COLUMN, BOOKING_DATE_COLUMN, LEAD_TIME_COLUMN, 'リードタイム', USAGE_COUNT_COLUMN
                        ]]
                        all_numeric_cols = data_for_modeling[potential_features].select_dtypes(include=['number']).columns.tolist()
                        all_category_cols = data_for_modeling[potential_features].select_dtypes(exclude=['number', 'datetime', 'timedelta']).columns.tolist()
                        ignored_numeric = list(set(all_numeric_cols) - set(selected_numeric))
                        ignored_categorical = list(set(all_category_cols) - set(selected_categorical))
                        # ★★★ '利用台数累積_lag30' を削除 ★★★
                        explicitly_ignored = ['曜日_name', 'en_name'] # lag30 は選択されなければ無視される
                        final_ignore_features = list(set(ignored_numeric + ignored_categorical + explicitly_ignored))

                        # モデル比較
                        best_model, comparison_results, setup_result = setup_and_compare_models(
                            _data=data_for_modeling, target=TARGET_VARIABLE,
                            numeric_features=selected_numeric, categorical_features=selected_categorical,
                            ignore_features=final_ignore_features, include_models=models_to_compare,
                            sort_metric='RMSE'
                        )

                        if best_model is not None and setup_result is not None:
                            st.markdown("---")
                            st.subheader(f"最良モデル ({type(best_model).__name__}) の特徴量重要度")
                            importance_df = get_feature_importance_df(best_model, setup_result)
                            if importance_df is not None and not importance_df.empty:
                                fig_importance = plot_feature_importance(importance_df)
                                if fig_importance: st.plotly_chart(fig_importance, use_container_width=True)
                                with st.expander("特徴量重要度データ"): st.dataframe(importance_df)
                            else: st.info("このモデルでは特徴量重要度を表示できません。")

                            # シナリオ予測
                            last_change_lt = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)
                            if last_change_lt is not None:
                                st.write(f"価格最終変更リードタイム: {last_change_lt}")
                                data_scenario = create_scenario_data(
                                    data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN,
                                    scenario_type='last_change_fixed', change_lead_time=last_change_lt
                                )
                                scenario_title_suffix = f"価格最終変更点(LT={last_change_lt})固定シナリオ"

                                if not data_scenario.empty:
                                    with st.expander(f"予測に使用したデータ ({scenario_title_suffix}) のサンプル"):
                                        # ★★★ 初期リストを selected_features のコピーで開始 ★★★
                                        display_columns = selected_features.copy()
                                        # 必須列を追加 (重複しないようにチェック)
                                        essential_columns = [LEAD_TIME_COLUMN, TARGET_VARIABLE] + PRICE_COLUMNS
                                        for col in essential_columns:
                                            if col in data_scenario.columns and col not in display_columns:
                                                display_columns.append(col)
                                        # ラグ列名を追加 (重複しないようにチェック)
                                        lag_col_name = f"{LAG_TARGET_COLUMN}_lag{LAG_DAYS}"
                                        # ★★★ 重複チェックを追加 ★★★
                                        if lag_col_name in data_scenario.columns and lag_col_name not in display_columns:
                                             display_columns.append(lag_col_name)

                                        # 存在する列のみにフィルタリング & 最終重複削除
                                        existing_display_columns = [col for col in display_columns if col in data_scenario.columns]
                                        if pd.Series(existing_display_columns).duplicated().any():
                                            existing_display_columns = pd.Series(existing_display_columns).drop_duplicates().tolist()

                                        if existing_display_columns:
                                            try:
                                                # ★★★ エラー箇所 ★★★
                                                st.dataframe(data_scenario[existing_display_columns].head())
                                                csv = data_scenario[existing_display_columns].to_csv(index=False).encode('utf-8')
                                                filename = f"scenario_data_{selected_date}_{selected_car_class}_{scenario_title_suffix.replace(' ', '_')}.csv"
                                                st.download_button("💾 予測用データをダウンロード", csv, filename, "text/csv")
                                            except ValueError as e:
                                                 st.error(f"シナリオデータサンプル表示中にエラー: {e}")
                                                 st.write("表示しようとした列:", existing_display_columns)
                                        else:
                                            st.warning("表示列なし")

                                    predictions = predict_with_model(best_model, data_scenario)
                                    if not predictions.empty:
                                        # 結果表示
                                        st.markdown("---")
                                        col_m1, col_m2 = st.columns(2)
                                        with col_m1:
                                            st.subheader("モデル評価比較結果")
                                            st.dataframe(comparison_results)
                                        with col_m2:
                                            st.subheader(f"実績 vs 予測比較 ({scenario_title_suffix}) - LT {last_change_lt} 以降")
                                            actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                            if LEAD_TIME_COLUMN not in predictions.columns:
                                                predictions_with_lt = pd.merge(predictions, data_scenario[[LEAD_TIME_COLUMN]], left_index=True, right_index=True, how='left')
                                            else:
                                                predictions_with_lt = predictions
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
                                                st.warning(f"LT {last_change_lt} 以降の表示データなし")
                                    else:
                                        st.error("予測実行失敗")
                                else:
                                    st.error("シナリオデータ作成失敗")
                            else:
                                st.warning("価格変動が見つからなかったため、最終価格固定シナリオでの予測は実行できません。")
                        elif best_model is None:
                            st.error("モデル比較失敗")
                        else:
                            st.error("PyCaretセットアップ失敗")
            else:
                st.warning(f"'{LEAD_TIME_COLUMN}'列なし")
        else:
             st.info(f"'{selected_date}' ({selected_car_class}) のデータなし")
    elif selected_date is None:
        # データがあり利用日列もあるが日付未選択の場合
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]) and not data[DATE_COLUMN].isnull().all():
             st.info("サイドバーから分析したい利用日を選択してください。")

# --- データ分析・修正ページ描画関数 ---
def render_data_analysis_page(data: pd.DataFrame):
    st.title("データ分析・修正")

    # --- 前回の更新サマリー表示 --- #
    if 'last_update_summary' in st.session_state and st.session_state['last_update_summary']:
        summary = st.session_state['last_update_summary']
        st.markdown("---")
        st.subheader("前回のデータ更新・再計算結果")
        if summary.get("status") == "success":
            # ★★★ Nullify と Lag の両方の結果を表示 ★★★
            nullify_res = summary.get("nullify_result", {})
            lag_res = summary.get("lag_recalc_result", {})

            success_messages = []
            if nullify_res.get("count") is not None: # count が 0 でも表示
                 if nullify_res.get("count", 0) > 0:
                     success_messages.append(
                         f"**{summary.get('date', '不明')}**以降の**{nullify_res.get('count', 0)}行**で列**{nullify_res.get('cols', [])}**をNAに更新。"
                     )
                 else:
                     success_messages.append(
                         f"**{summary.get('date', '不明')}**以降のデータ更新対象なし。"
                     )

            if lag_res:
                 success_messages.append(
                     f"ラグ特徴量**'{lag_res.get('lag_col_name', '?')}'**を再計算({lag_res.get('nan_count', '?')}行NaN)。"
                 )

            if success_messages:
                 st.success("✅ " + " ".join(success_messages))
            else:
                 st.info("ℹ️ データ更新・再計算処理は実行されましたが、変更はありませんでした。")

        else: # status が 'error' または不明な場合
             st.warning(f"⚠️ 前回の処理で問題が発生: {summary.get('message', '不明なエラー')}")

        del st.session_state['last_update_summary']
        st.markdown("---")

    # --- サイドバーウィジェット ---
    selected_analysis_date, analyze_button = render_data_analysis_sidebar_widgets(data)

    # --- メインエリア --- #
    st.header("データ探索")
    display_exploration(data)
    st.markdown("---")

    st.header(f"特定予約日以降の '{USAGE_COUNT_COLUMN}' 日別合計推移")
    st.write(f"指定した日付より**後**の予約日について、日ごとの '{USAGE_COUNT_COLUMN}' の合計値をグラフ表示します。")

    # --- 日別合計グラフ表示 & ゼロ日付特定 --- #
    if selected_analysis_date is not None:
        daily_sum_df = analyze_daily_sum_after_date(data=data, start_date=selected_analysis_date, booking_date_col=BOOKING_DATE_COLUMN, sum_col=USAGE_COUNT_COLUMN)
        if daily_sum_df is not None:
            if not daily_sum_df.empty:
                st.line_chart(daily_sum_df)
                zero_date_str = None
                daily_sum_df_sorted = daily_sum_df.sort_index()
                sum_col_name = f'{USAGE_COUNT_COLUMN}_合計'
                if sum_col_name in daily_sum_df_sorted.columns:
                     zero_rows = daily_sum_df_sorted[daily_sum_df_sorted[sum_col_name] <= 0]
                     if not zero_rows.empty:
                         zero_date_str = zero_rows.index[0].strftime('%Y-%m-%d')
                         st.info(f"📈グラフの値が最初に0になった日付: **{zero_date_str}**")
                         st.session_state['zero_cutoff_date'] = zero_date_str
                     else:
                         st.info("📈グラフ期間中に値が0になる日はありませんでした。")
                         if 'zero_cutoff_date' in st.session_state: del st.session_state['zero_cutoff_date']
                else: st.warning("グラフデータから合計列が見つかりませんでした。")
                with st.expander("詳細データ表示"):
                     st.dataframe(daily_sum_df)
    else:
        st.info("サイドバーで起点となる予約日を選択してください。")

    # --- データ更新とラグ再計算セクション --- #
    st.markdown("---")
    st.header("データ更新とラグ特徴量再計算")

    if 'zero_cutoff_date' in st.session_state and st.session_state['zero_cutoff_date']:
        cutoff_date_str = st.session_state['zero_cutoff_date']
        st.write(f"特定された日付 **{cutoff_date_str}** 以降のデータについて、")
        st.write(f"'{USAGE_COUNT_COLUMN}' および '{TARGET_VARIABLE}' を欠損値(NA)に更新し、")
        st.write(f"'{LAG_TARGET_COLUMN}_lag{LAG_DAYS}' を再計算します。")

        update_button = st.button(f"🔄 {cutoff_date_str} 以降のデータを更新＆ラグ再計算", key="update_data_button")

        if update_button:
            update_status = "error"
            update_message = ""
            nullify_result = None # Nullify結果用
            lag_recalc_result = None # Lag計算結果用

            with st.spinner("データ更新とラグ再計算を実行中..."):
                try:
                    current_data = st.session_state.get('processed_data')
                    if current_data is None:
                         update_message = "セッションからデータが見つかりません。"
                         # エラーメッセージは最後にまとめてサマリーに格納
                         st.error(f"エラー: {update_message}") # ここでは表示しておく
                         st.stop()

                    cutoff_date = pd.to_datetime(cutoff_date_str).date()
                    cols_to_null = [USAGE_COUNT_COLUMN, TARGET_VARIABLE]

                    # 1. データ更新
                    data_nulled, num_rows_updated, updated_cols = nullify_usage_data_after_date(
                        df=current_data.copy(),
                        cutoff_date=cutoff_date,
                        date_col=BOOKING_DATE_COLUMN,
                        cols_to_nullify=cols_to_null
                    )
                    # Nullify結果を保存
                    nullify_result = {"count": num_rows_updated, "cols": updated_cols}

                    if data_nulled is not None and num_rows_updated is not None:
                        # 2. ラグ特徴量再計算
                        # st.info("ラグ特徴量を再計算中...") # 冗長なのでコメントアウト
                        data_recalculated, lag_info = recalculate_lag_feature(
                            df_processed=data_nulled,
                            lag_target_col=LAG_TARGET_COLUMN,
                            lag_days=LAG_DAYS,
                            booking_date_col=BOOKING_DATE_COLUMN,
                            group_cols=LAG_GROUP_COLS
                        )
                        # Lag計算結果を保存
                        lag_recalc_result = lag_info

                        if data_recalculated is not None and isinstance(data_recalculated, pd.DataFrame):
                            st.session_state['processed_data'] = data_recalculated
                            update_status = "success"
                        else:
                            update_message = "ラグ特徴量の再計算に失敗しました。"
                    else:
                         update_message = "データ更新処理に失敗しました。"

                except Exception as e_update:
                    update_message = f"予期せぬエラー: {e_update}"
                    # エラー発生時もサマリーに情報を残す
                    st.error(f"データ更新またはラグ再計算中にエラーが発生しました: {update_message}")

                # --- 結果をセッションステートに保存 --- #
                st.session_state['last_update_summary'] = {
                    "status": update_status,
                    "message": update_message,
                    "date": cutoff_date_str,
                    "nullify_result": nullify_result,
                    "lag_recalc_result": lag_recalc_result
                }

                if 'zero_cutoff_date' in st.session_state:
                    del st.session_state['zero_cutoff_date']
                # st.rerun() は削除済み

            # ★★★ ボタン処理の最後に st.rerun() を追加 ★★★
            # これにより、スピナーが消え、ページ上部のサマリー表示が更新される
            st.rerun()

    else:
        st.info("先に上記の「日別合計推移」の分析を実行し、グラフが0になる日付を特定してください。")

# --- アプリケーション本体 (main関数) ---
def main():
    st.set_page_config(layout="wide")
    config = load_config()

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
            ("予測・比較分析", "データ分析・修正"),
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
        if app_mode == "予測・比較分析":
            render_prediction_analysis_page(current_data, config)
        elif app_mode == "データ分析・修正":
            render_data_analysis_page(current_data)
        else:
            st.error("無効なモードが選択されました。")
    elif uploaded_file is None:
         st.info("サイドバーから分析対象のCSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()