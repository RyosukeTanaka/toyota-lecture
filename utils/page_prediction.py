# utils/page_prediction.py

import streamlit as st
import pandas as pd
from .constants import ( # constants からインポート
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN,
    LAG_TARGET_COLUMN, LAG_DAYS
)
from .data_processing import ( # 相対インポートに変更
    filter_data_by_date, create_scenario_data,
    find_last_price_change_lead_time
)
from .visualization import ( # 相対インポートに変更
    plot_booking_curve, plot_price_trends, plot_comparison_curve,
    plot_feature_importance
)
from .modeling import ( # 相対インポートに変更
    setup_and_compare_models, predict_with_model, get_feature_importance_df
)
from .ui_components import ( # 相対インポートに変更
    render_prediction_sidebar_widgets
)

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

                                    # ★★★ predict_with_model の戻り値をアンパック ★★★
                                    predictions, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation = predict_with_model(best_model, data_scenario, target=TARGET_VARIABLE)

                                    # ★★★ 補完ログがあればテーブルで表示 ★★★
                                    if imputation_log:
                                        st.subheader("予測前の特徴量欠損値補完の詳細")
                                        log_df = pd.DataFrame(imputation_log)
                                        if 'Imputation Value' in log_df.columns:
                                             log_df['Imputation Value'] = log_df['Imputation Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                                        st.dataframe(log_df)
                                        # ★★★ NaNがあった行も表示 ★★★
                                        if nan_rows_before_imputation is not None and not nan_rows_before_imputation.empty:
                                            st.subheader("NaN値が含まれていた行（補完前）")
                                            st.dataframe(nan_rows_before_imputation)
                                        # ★★★ 補完後のNaNがあった行も表示 ★★★
                                        if nan_rows_after_imputation is not None and not nan_rows_after_imputation.empty:
                                            st.subheader("NaN値が含まれていた行（補完後）")
                                            st.dataframe(nan_rows_after_imputation)
                                        # ★★★ ------------------------- ★★★
                                        st.markdown("---") # テーブルと結果の間に区切り
                                    # ★★★ ---------------------------- ★★★

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