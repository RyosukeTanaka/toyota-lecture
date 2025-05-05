# utils/page_prediction.py

import streamlit as st
import pandas as pd
from typing import Dict, Any
import datetime
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
from .model_storage import load_model, list_saved_models, load_comparison_results

# --- 予測・比較分析ページ描画関数 ---
def render_prediction_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    st.title("予測分析")

    # --- サイドバーウィジェットの描画と値の取得 ---
    (
        selected_car_class,
        selected_date,
        selected_model_info,
        run_prediction
    ) = render_prediction_sidebar_widgets(data)

    # --- メインエリア --- #
    st.subheader("処理済みデータプレビュー (現在の状態)")
    st.dataframe(data.head())
    # データ探索は別ページ

    st.markdown("---") # プレビューと結果の間に区切り

    if not selected_model_info:
        st.warning("予測を実行するには、まず「モデルトレーニング」ページでモデルを作成し、サイドバーで使用するモデルを選択してください。")
        return

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

                # モデル比較結果（サイドバーに表示したモデル情報とは別に、詳細情報をエクスパンダーで表示）
                comparison_path = selected_model_info.get("comparison_results_path")
                if comparison_path:
                    with st.expander("モデル比較結果の詳細"):
                        comparison_results = load_comparison_results(comparison_path)
                        if comparison_results is not None:
                            st.dataframe(comparison_results)
                        else:
                            st.warning("比較結果を読み込めませんでした。")

                # 予測実行セクション
                if run_prediction:
                    st.markdown("---")
                    st.header("予測実行")
                    with st.spinner('予測を実行中...'):
                        # モデルのロード
                        model = load_model(selected_model_info["path"])
                        
                        if model is None:
                            st.error("モデルの読み込みに失敗しました。")
                            return

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
                                    # selected_featuresをモデルのメタデータから取得
                                    selected_features = selected_model_info.get("selected_features", [])
                                    display_columns = selected_features.copy() if selected_features else []
                                    
                                    # 必須列を追加 (重複しないようにチェック)
                                    essential_columns = [LEAD_TIME_COLUMN, TARGET_VARIABLE] + PRICE_COLUMNS
                                    for col in essential_columns:
                                        if col in data_scenario.columns and col not in display_columns:
                                            display_columns.append(col)
                                    
                                    # ラグ列名を追加 (重複しないようにチェック)
                                    lag_col_name = f"{LAG_TARGET_COLUMN}_lag{LAG_DAYS}"
                                    if lag_col_name in data_scenario.columns and lag_col_name not in display_columns:
                                         display_columns.append(lag_col_name)

                                    # 存在する列のみにフィルタリング & 最終重複削除
                                    existing_display_columns = [col for col in display_columns if col in data_scenario.columns]
                                    if pd.Series(existing_display_columns).duplicated().any():
                                        existing_display_columns = pd.Series(existing_display_columns).drop_duplicates().tolist()

                                    if existing_display_columns:
                                        try:
                                            st.dataframe(data_scenario[existing_display_columns].head())
                                            csv = data_scenario[existing_display_columns].to_csv(index=False).encode('utf-8')
                                            filename = f"scenario_data_{selected_date}_{selected_car_class}_{scenario_title_suffix.replace(' ', '_')}.csv"
                                            st.download_button("💾 予測用データをダウンロード", csv, filename, "text/csv")
                                        except ValueError as e:
                                             st.error(f"シナリオデータサンプル表示中にエラー: {e}")
                                             st.write("表示しようとした列:", existing_display_columns)
                                    else:
                                        st.warning("表示列なし")

                                # 予測実行
                                predictions, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation = predict_with_model(model, data_scenario, target=TARGET_VARIABLE)

                                # 補完ログがあればテーブルで表示
                                if imputation_log:
                                    st.subheader("予測前の特徴量欠損値補完の詳細")
                                    log_df = pd.DataFrame(imputation_log)
                                    if 'Imputation Value' in log_df.columns:
                                         log_df['Imputation Value'] = log_df['Imputation Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                                    st.dataframe(log_df)
                                    
                                    if nan_rows_before_imputation is not None and not nan_rows_before_imputation.empty:
                                        st.subheader("NaN値が含まれていた行（補完前）")
                                        st.dataframe(nan_rows_before_imputation)
                                    
                                    if nan_rows_after_imputation is not None and not nan_rows_after_imputation.empty:
                                        st.subheader("NaN値が含まれていた行（補完後）")
                                        st.dataframe(nan_rows_after_imputation)
                                    
                                    st.markdown("---") # テーブルと結果の間に区切り

                                if not predictions.empty:
                                    # 結果表示
                                    st.markdown("---")
                                    
                                    # 実績 vs 予測の比較グラフと表
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
            else:
                st.warning(f"'{LEAD_TIME_COLUMN}'列なし")
        else:
             st.info(f"'{selected_date}' ({selected_car_class}) のデータなし")
    elif selected_date is None:
        # データがあり利用日列もあるが日付未選択の場合
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]) and not data[DATE_COLUMN].isnull().all():
             st.info("サイドバーから分析したい利用日を選択してください。") 