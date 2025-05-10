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
    plot_feature_importance, plot_full_period_comparison
)
from .modeling import ( # 相対インポートに変更
    setup_and_compare_models, predict_with_model, get_feature_importance_df
)
from .ui_components import ( # 相対インポートに変更
    render_prediction_sidebar_widgets
)
from .model_storage import load_model, list_saved_models, load_comparison_results, get_model_metadata, prepare_features_for_prediction
import numpy as np
from .revenue_analysis import calculate_revenue_difference, plot_revenue_comparison

# --- 予測・比較分析ページ描画関数 ---
def render_prediction_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    st.title("予測分析")

    # --- サイドバーウィジェットの描画と値の取得 ---
    (
        selected_car_class,
        selected_model_info,
        _  # 予測実行ボタンはメインエリアで表示するため無視
    ) = render_prediction_sidebar_widgets(data)

    # --- メインエリア --- #
    if not selected_model_info:
        st.warning("予測を実行するには、まず「モデルトレーニング」ページでモデルを作成し、サイドバーで使用するモデルを選択してください。")
        return

    # モデル比較結果の詳細セクション（独立したセクションとして表示）
    st.header("モデル比較結果の詳細")
    comparison_path = selected_model_info.get("comparison_results_path")
    if comparison_path:
        comparison_results = load_comparison_results(comparison_path)
        if comparison_results is not None:
            st.dataframe(comparison_results)
        else:
            st.warning("比較結果を読み込めませんでした。")
    else:
        st.info("このモデルには比較結果が保存されていません。")
    
    st.markdown("---") # 比較結果と次のセクションの間に区切り線

    # --- 分析日の選択（モデル比較結果の後に表示） ---
    st.header("利用日の選択")
    
    # 選択された車両クラスでデータをフィルタリング (日付選択用)
    if selected_car_class == "全クラス":
        data_for_date_selection = data
    else:
        data_for_date_selection = data[data[CAR_CLASS_COLUMN] == selected_car_class]

    # 利用日の選択ウィジェット
    selected_date = None
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
        else:
            st.info(f"'{selected_car_class}'クラスには有効な'{DATE_COLUMN}'データがありません。")
    else:
        st.warning(f"'{DATE_COLUMN}'列がないか日付型ではありません。")
        selected_date = None # Noneにリセット

    # 予測実行ボタン（モデルと利用日の両方が選択された場合に表示）
    run_prediction = False
    if selected_model_info and selected_date:
        run_prediction = st.button("🔮 予測実行", key="run_prediction")
    
    st.markdown("---") # 日付選択と次のセクションの間に区切り線

    # 処理済みデータプレビュー（日付選択の後に表示）
    st.subheader("処理済みデータプレビュー (現在の状態)")
    st.dataframe(data.head())

    st.markdown("---") # データプレビューと次のセクションの間に区切り線

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

                # --- last_change_lt と 価格_トヨタの変動チェックを先に行う ---
                last_change_lt = None
                has_toyota_price_changed = False
                toyota_price_column_name = PRICE_COLUMNS[0] if PRICE_COLUMNS else None # PRICE_COLUMNSが空でないか確認

                if toyota_price_column_name and toyota_price_column_name in data_filtered_sorted.columns:
                    has_toyota_price_changed = data_filtered_sorted[toyota_price_column_name].nunique(dropna=True) > 1
                
                # last_change_lt の計算 (変動があった列のみが対象となる)
                last_change_lt = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)

                # --- 予測実行ボタンの表示条件 ---
                run_prediction_disabled = True # デフォルトは無効
                button_message = None

                if not selected_model_info:
                    button_message = "モデルが選択されていません。" # これは既にサイドバーで警告されるはず
                elif not has_toyota_price_changed and toyota_price_column_name:
                    button_message = f"'{toyota_price_column_name}' に価格変動がありませんでした。価格固定シナリオでの予測は、価格変動があった場合に特に有効です。"
                    st.warning(button_message)
                elif last_change_lt is None:
                    button_message = "主要な価格列に価格変動が見られませんでした。価格固定シナリオでの予測は実行できません。"
                    st.warning(button_message)
                else:
                    run_prediction_disabled = False # 有効化

                # 実際の予約曲線と価格推移グラフ
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("実際の予約曲線")
                    fig_actual = plot_booking_curve(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_col=TARGET_VARIABLE, title=f"{selected_date} {selected_car_class} 実際の予約曲線")
                    st.plotly_chart(fig_actual, use_container_width=True)

                    # テーブル表示とダウンロードボタンを追加
                    with st.expander("グラフデータ詳細"):
                        st.dataframe(data_filtered_sorted)
                        csv_actual = data_filtered_sorted.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 実績データをダウンロード",
                            data=csv_actual,
                            file_name=f"actual_booking_data_{selected_date}_{selected_car_class}.csv",
                            mime="text/csv",
                        )
                with col2:
                    st.subheader("価格推移")
                    fig_prices = plot_price_trends(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_cols=PRICE_COLUMNS, title=f"{selected_date} {selected_car_class} 価格推移")
                    st.plotly_chart(fig_prices, use_container_width=True)

                # 予測実行セクション
                if run_prediction:
                    # まず、このページで計算した last_change_lt と has_toyota_price_changed に基づく実行可否を再確認
                    can_run_scenario_prediction = True
                    scenario_error_message = None

                    if not has_toyota_price_changed and toyota_price_column_name:
                        scenario_error_message = f"'{toyota_price_column_name}' に価格変動がありませんでした。価格固定シナリオでの予測は、価格変動があった場合に特に有効です。"
                        can_run_scenario_prediction = False
                    elif last_change_lt is None:
                        scenario_error_message = "主要な価格列に価格変動が見られませんでした。価格固定シナリオでの予測は実行できません。"
                        can_run_scenario_prediction = False
                    
                    if not can_run_scenario_prediction:
                        st.error(f"予測は実行できませんでした: {scenario_error_message}")
                    else:
                        # このelseブロックに到達するのは、last_change_lt が有効な値を持っている場合
                        st.markdown("---")
                        st.header("予測実行")
                        with st.spinner('予測を実行中...'):
                            # モデルのロード
                            model = load_model(selected_model_info["path"])
                            
                            if model is None:
                                st.error("モデルの読み込みに失敗しました。")
                                st.stop() # アプリの実行をここで停止
                            
                            # last_change_lt は既に計算済みで、Noneでないことが保証されている
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

                                # モデルのメタデータを取得
                                model_filename = selected_model_info.get("filename")
                                model_metadata = None
                                
                                # デバッグログをエクスパンダーに移動
                                with st.expander("予測プロセスのデバッグ情報", expanded=False):
                                    st.info("モデルの内部構造から特徴量情報を検査中...")
                                    
                                    # *** 新しい方法: モデル自体から特徴量名を直接取得する ***
                                    try:
                                        # スキップされる可能性がある一般的なモデル・パイプライン属性
                                        skipped_attrs = ['_isfit', 'classes_', 'n_classes_', 'n_features_in_', 'base_score', 'label_encoder', 'estimator']
                                        
                                        # モデル属性の検査
                                        model_attrs = dir(model)
                                        feature_attrs = [attr for attr in model_attrs 
                                                       if 'feature' in attr.lower() and attr not in skipped_attrs]
                                        
                                        # 特徴量名を検出
                                        model_features = None
                                        if hasattr(model, 'feature_names_in_'):
                                            model_features = list(model.feature_names_in_)
                                            st.success(f"モデルから直接 {len(model_features)}個の特徴量名を取得しました (feature_names_in_)")
                                        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                                            # XGBoostの場合
                                            model_features = model.get_booster().feature_names
                                            st.success(f"XGBoostモデルから {len(model_features)}個の特徴量名を取得しました")
                                        elif hasattr(model, 'steps'):
                                            # パイプラインの場合は最後のステップを確認
                                            final_step = model.steps[-1][1]
                                            if hasattr(final_step, 'feature_names_in_'):
                                                model_features = list(final_step.feature_names_in_)
                                                st.success(f"パイプラインの最終ステップから {len(model_features)}個の特徴量名を取得しました")
                                        
                                        # メタデータの特徴量名を上書き
                                        if model_features:
                                            if model_filename:
                                                model_metadata = get_model_metadata(model_filename)
                                                if model_metadata:
                                                    # 元のmodel_columnsを保存
                                                    original_columns = model_metadata.get("model_columns", [])
                                                    if original_columns:
                                                        st.info(f"メタデータの特徴量名 ({len(original_columns)}個) をモデルから取得した特徴量名で上書きします")
                                                    
                                                    # model_featuresで上書き
                                                    model_metadata["model_columns"] = model_features
                                        
                                        # モデル構造のその他の情報を表示
                                        st.info(f"モデルの型: {type(model).__name__}")
                                        if feature_attrs:
                                            st.info(f"検出された特徴量関連の属性: {feature_attrs}")
                                    
                                    except Exception as e:
                                        st.warning(f"モデルの内部構造検査中にエラー: {e}")
                                    
                                    if not model_metadata and model_filename:
                                        model_metadata = get_model_metadata(model_filename)
                                    
                                    if model_metadata:
                                        if "model_columns" in model_metadata:
                                            st.success(f"特徴量情報が存在します: {len(model_metadata['model_columns'])}個の列")
                                            
                                            # 特徴量情報の例を表示
                                            st.subheader("特徴量情報サンプル")
                                            st.json(model_metadata["model_columns"][:10])
                                        else:
                                            st.warning("特徴量情報が見つかりません。予測が失敗する可能性があります。")
                                
                                # 予測時のフォールバック方法: ダイレクト予測を試みる
                                try:
                                    # 予測プロセスの開始を通知
                                    st.info("予測処理を開始します...")
                                    
                                    # デバッグログをエクスパンダーに移動
                                    with st.expander("予測実行詳細ログ", expanded=False):
                                        st.info("モデルのpredict()メソッドを直接使用して予測を実行します...")
                                        
                                        # 日付列を文字列に変換（最終手段）
                                        scen_data_transformed = data_scenario.copy()
                                        date_cols = scen_data_transformed.select_dtypes(include=['datetime64']).columns
                                        for col in date_cols:
                                            scen_data_transformed[col] = scen_data_transformed[col].dt.strftime('%Y-%m-%d')
                                        
                                        # ターゲット変数がある場合は除去
                                        if TARGET_VARIABLE in scen_data_transformed.columns:
                                            X = scen_data_transformed.drop(columns=[TARGET_VARIABLE])
                                        else:
                                            X = scen_data_transformed
                                        
                                        # 直接予測
                                        y_pred = None
                                        
                                        # 1. 通常のpredictを試行
                                        try:
                                            st.info("モデルの直接予測を試行...")
                                            if hasattr(model, 'predict'):
                                                y_pred = model.predict(X)
                                                st.success("直接予測成功!")
                                        except Exception as e1:
                                            st.error(f"直接予測でエラー: {e1}")
                                            
                                            # 2. 特徴量変換を実行して再度試行
                                            try:
                                                if model_metadata and "model_columns" in model_metadata:
                                                    st.info("特徴量変換を適用して再試行...")
                                                    transformed_data = prepare_features_for_prediction(X, model_metadata)
                                                    y_pred = model.predict(transformed_data)
                                                    st.success("特徴量変換後の予測成功!")
                                            except Exception as e2:
                                                st.error(f"特徴量変換後の予測でもエラー: {e2}")
                                    
                                    # 予測結果を利用
                                    if y_pred is not None:
                                        st.success("予測完了!")
                                        
                                        # 結果をデータフレームに変換
                                        predictions_result = data_scenario.copy()
                                        predictions_result['prediction_label'] = y_pred
                                        
                                        # 比較グラフと表を表示
                                        st.markdown("---")
                                        st.subheader(f"実績 vs 予測比較 ({scenario_title_suffix})")
                                        
                                        actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                        predictions_filtered_display = predictions_result[predictions_result[LEAD_TIME_COLUMN] <= last_change_lt]
                                        
                                        if not actual_filtered_display.empty and not predictions_filtered_display.empty:
                                            # タブを作成
                                            compare_tab1, compare_tab2 = st.tabs(["価格変更後のみ表示", "全期間比較表示"])
                                            
                                            with compare_tab1:
                                                # 価格変更後のみの比較グラフ
                                                fig_compare = plot_comparison_curve(
                                                    df_actual=actual_filtered_display, 
                                                    df_predicted=predictions_filtered_display,
                                                    x_col=LEAD_TIME_COLUMN, 
                                                    y_actual_col=TARGET_VARIABLE, 
                                                    y_pred_col='prediction_label',
                                                    title=f"{selected_date} {selected_car_class} 実績 vs 予測 (LT {last_change_lt} 以降)"
                                                )
                                                st.plotly_chart(fig_compare, use_container_width=True)
                                                
                                                st.subheader(f"実績 vs 予測 データテーブル (LT {last_change_lt} 以降)")
                                                df_actual_for_table = actual_filtered_display[[LEAD_TIME_COLUMN, TARGET_VARIABLE]].rename(columns={TARGET_VARIABLE: '実績利用台数'})
                                                df_pred_for_table = predictions_filtered_display[[LEAD_TIME_COLUMN, 'prediction_label']].rename(columns={'prediction_label': '予測利用台数'})
                                                df_comparison_table = pd.merge(df_actual_for_table, df_pred_for_table, on=LEAD_TIME_COLUMN, how='inner')
                                                st.dataframe(df_comparison_table.sort_values(by=LEAD_TIME_COLUMN).reset_index(drop=True))
                                            
                                            with compare_tab2:
                                                # 全期間の比較グラフ
                                                # データがフル期間について揃っているか確認
                                                if LEAD_TIME_COLUMN not in predictions_result.columns:
                                                    predictions_full = pd.merge(predictions_result, data_scenario[[LEAD_TIME_COLUMN]], left_index=True, right_index=True, how='left')
                                                else:
                                                    predictions_full = predictions_result
                                                
                                                fig_full_compare = plot_full_period_comparison(
                                                    df_actual=data_filtered_sorted,
                                                    df_predicted=predictions_full,
                                                    x_col=LEAD_TIME_COLUMN,
                                                    y_actual_col=TARGET_VARIABLE,
                                                    y_pred_col='prediction_label',
                                                    title=f"{selected_date} {selected_car_class} 実績 vs 予測 (全期間)",
                                                    change_lead_time=last_change_lt
                                                )
                                                st.plotly_chart(fig_full_compare, use_container_width=True)
                                                
                                                st.info("全期間の予約曲線を表示しています。緑色の縦線は価格最終変更点を示しています。")
                                                st.markdown("**注意**: 価格変更点より前（右側）は実際の価格変動に基づく実績データですが、予測は価格固定シナリオに基づいています。")
                                        else:
                                            st.warning(f"LT {last_change_lt} 以降の表示データなし")
                                            
                                        # 売上比較分析
                                        st.markdown("---")
                                        st.subheader("売上金額ベースでの比較分析")
                                        
                                        # 売上差額の計算
                                        st.info("価格変更がもたらした売上への影響を分析しています...")
                                        (revenue_df, total_actual, total_predicted, total_difference,
                                         actual_before, actual_after, predicted_after
                                        ) = calculate_revenue_difference(
                                            df_actual=data_filtered_sorted,
                                            df_predicted=predictions_result,
                                            lead_time_col=LEAD_TIME_COLUMN,
                                            actual_usage_col=TARGET_VARIABLE,
                                            pred_usage_col='prediction_label',
                                            price_col=PRICE_COLUMNS[0], # 価格_トヨタを使用
                                            change_lead_time=last_change_lt
                                        )
                                        
                                        if not revenue_df.empty:
                                            # 売上差額メトリクス表示
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("実績総売上", f"{int(total_actual):,}円")
                                            with col2:
                                                st.metric("予測総売上（価格固定）", f"{int(total_predicted):,}円")
                                            with col3:
                                                delta_color = "normal" if total_difference >= 0 else "inverse"
                                                st.metric("売上差額（実績-予測）", f"{int(total_difference):,}円", delta=f"{int(total_difference):,}円", delta_color=delta_color)
                                            
                                            # 売上推移グラフ
                                            st.subheader("売上金額推移")
                                            fig_revenue = plot_revenue_comparison(
                                                revenue_df=revenue_df,
                                                lead_time_col=LEAD_TIME_COLUMN,
                                                title=f"{selected_date} {selected_car_class} 売上金額比較 (LT {last_change_lt} 以降)"
                                            )
                                            st.plotly_chart(fig_revenue, use_container_width=True)
                                            
                                            # 売上詳細テーブル
                                            with st.expander("売上計算詳細データ"):
                                                st.dataframe(revenue_df.sort_values(by=LEAD_TIME_COLUMN, ascending=False))
                                                # CSVダウンロード機能
                                                csv = revenue_df.to_csv(index=False).encode('utf-8')
                                                filename = f"revenue_analysis_{selected_date}_{selected_car_class}.csv"
                                                st.download_button("💾 売上分析データをダウンロード", csv, filename, "text/csv")
                                            
                                            # 結果の解釈
                                            if total_difference > 0:
                                                st.success(f"**分析結果**: 価格変更により **{int(total_difference):,}円** の追加売上が発生したと推定されます。価格戦略が有効に機能しています。")
                                            elif total_difference < 0:
                                                st.warning(f"**分析結果**: 価格変更により **{abs(int(total_difference)):,}円** の売上減少があったと推定されます。価格戦略の見直しが必要かもしれません。")
                                            else:
                                                st.info("**分析結果**: 価格変更による売上への顕著な影響は見られませんでした。")
                                        else:
                                            st.warning("売上計算に必要なデータが不足しているため、売上分析を表示できません。")
                                except Exception as e_pred_main: # 例外変数名を変更
                                    st.error(f"予測処理のメインロジックでエラーが発生しました: {e_pred_main}")
                                    # 詳細なエラー情報をログやデバッグ出力に追加することを検討
                                    st.exception(e_pred_main) # トレースバックも表示
                                    
                                    # 従来の方法をエクスパンダーに移動
                                    with st.expander("フォールバック予測方法の詳細ログ", expanded=False):
                                        st.warning("フォールバック予測方法を試行します...")
                            else:
                                st.error("シナリオデータ作成失敗")
            else:
                st.warning(f"'{LEAD_TIME_COLUMN}'列なし")
        else:
             st.info(f"'{selected_date}' ({selected_car_class}) のデータなし")
    elif selected_date is None:
        # 日付選択フィールドの存在確認
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]) and not data[DATE_COLUMN].isnull().all():
             st.info("分析結果を表示するには、利用日を選択してください。")
        else:
            st.warning("日付選択に問題があります。") 