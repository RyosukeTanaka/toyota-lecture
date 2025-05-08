# utils/page_batch_analysis.py

import streamlit as st
import pandas as pd
import datetime
import os
from typing import Dict, Any, List, Optional, Tuple
import concurrent.futures
import plotly.io as pio
import numpy as np

from .constants import (
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN
)
from .ui_components import render_prediction_sidebar_widgets
from .model_storage import load_model, get_model_metadata, list_saved_models
from .batch_analysis import run_batch_prediction
from .visualization import plot_batch_revenue_comparison, plot_price_change_vs_booking_impact

# ---- バックグラウンドで画像を保存するヘルパー関数 (一時的に処理をコメントアウト) ----
def _save_image_task(fig: Any, filepath: str):
    """Plotlyの図を画像ファイルとして保存する（バックグラウンドタスク用）"""
    # try:
    #     pio.write_image(fig, filepath)
    #     # print(f"画像保存完了: {filepath}") # デバッグ用コンソール出力
    #     return f"Successfully saved {filepath}"
    # except Exception as e:
    #     # print(f"画像保存エラー {filepath}: {e}") # デバッグ用コンソール出力
    #     return f"Error saving {filepath}: {e}"
    st.warning(f"画像保存処理は一時的にスキップされました: {filepath}")
    return f"Image saving skipped for {filepath}" # スキップしたことを示すメッセージ

def save_batch_results_to_folder(
    metadata_list: Optional[List[Dict[str, Any]]],
    date_revenue_df: Optional[pd.DataFrame],
    class_revenue_df: Optional[pd.DataFrame],
    result_df: Optional[pd.DataFrame],
    fig_date: Optional[Any],
    fig_class: Optional[Any]
) -> Optional[str]:
    """バッチ分析結果をローカルフォルダに保存する"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"batch_results_{timestamp}"
    try:
        if not os.path.exists(results_dir): os.makedirs(results_dir)
    except OSError as e:
        st.error(f"結果保存用フォルダの作成に失敗しました: {results_dir} ({e})")
        return None

    try:
        if result_df is not None and not result_df.empty:
            expected_cols = [
                 "利用日", "車両クラス", "使用モデル", "価格変更リードタイム",
                 "利用台数累積実績（価格変更時）",
                 "利用台数累積実績",
                 "利用台数累積（予測）",
                 "追加実績予約数（価格変更後）",
                 "追加予測予約数（価格変更後）",
                 "変更前価格", "変更後価格",
                 "実績売上（変更前）",
                 "実績売上（変更後）",
                 "予測売上（変更後）",
                 "実績売上",
                 "予測売上（価格変更影響なし時は実績値）",
                 "売上差額"
            ]
            
            cols_present = [col for col in expected_cols if col in result_df.columns]
            cols_missing = [col for col in expected_cols if col not in result_df.columns]
            
            if cols_missing:
                st.warning(f"CSV保存警告: 以下の列が結果DataFrameに存在しませんでした: {cols_missing}", icon="⚠️")

            if cols_present:
                st.info(f"CSV保存: 保存する列: {cols_present}")
                result_df[cols_present].to_csv(
                    os.path.join(results_dir, "batch_analysis_results.csv"),
                    index=False,
                    float_format='%.1f'
                )
            else:
                 st.warning("CSV保存警告: 保存対象の列が結果DataFrameにありませんでした。batch_analysis_results.csv は空になります。", icon="⚠️")

        if date_revenue_df is not None and not date_revenue_df.empty: date_revenue_df.to_csv(os.path.join(results_dir, "date_revenue_summary.csv"), index=False, encoding='utf-8-sig')
        if class_revenue_df is not None and not class_revenue_df.empty: class_revenue_df.to_csv(os.path.join(results_dir, "class_revenue_summary.csv"), index=False, encoding='utf-8-sig')

    except Exception as e: st.warning(f"CSVファイルの保存中にエラーが発生しました: {e}")
    
    try:
        if fig_date: fig_date.write_html(os.path.join(results_dir, "date_revenue_chart.html"))
        if fig_class: fig_class.write_html(os.path.join(results_dir, "class_revenue_chart.html"))
    except Exception as e: st.warning(f"HTMLグラフの保存中にエラー: {e}")

    image_tasks = []
    if fig_date:
        image_tasks.append((fig_date, os.path.join(results_dir, "date_revenue_chart.png")))
    if fig_class:
        image_tasks.append((fig_class, os.path.join(results_dir, "class_revenue_chart.png")))

    if image_tasks:
        st.warning("PNG画像の保存処理は、問題調査のため一時的に無効化されています。")
        for fig, fp in image_tasks:
            _save_image_task(fig, fp)

    try:
        if metadata_list:
            success_count = sum(1 for meta in metadata_list if meta.get("success", False))
            fail_count = len(metadata_list) - success_count
            success_data = [meta for meta in metadata_list if meta.get("success", False)]
            total_actual = sum(meta.get("revenue_actual", 0) for meta in success_data)
            total_predicted = sum(meta.get("revenue_predicted", 0) for meta in success_data)
            total_difference = sum(meta.get("revenue_difference", 0) for meta in success_data)
            
            with open(os.path.join(results_dir, "summary.txt"), "w", encoding='utf-8') as f:
                f.write(f"バッチ分析サマリー\n")
                f.write(f"実行日時: {timestamp}\n")
                f.write(f"=================================\n")
                f.write(f"処理総数: {len(metadata_list)}件\n")
                f.write(f"成功: {success_count}件, 失敗: {fail_count}件\n\n")
                f.write(f"売上集計結果:\n")
                f.write(f"実績総売上: {int(total_actual):,}円\n")
                f.write(f"予測売上（価格変更影響なし時は実績値）: {int(total_predicted):,}円\n")
                f.write(f"売上差額（実績-予測）: {int(total_difference):,}円\n\n")
                if total_difference > 0:
                    f.write(f"全体分析: 期間全体で価格変更により {int(total_difference):,}円 の追加売上が発生したと推定されます。価格戦略は有効に機能しています。\n")
                elif total_difference < 0:
                    f.write(f"全体分析: 期間全体で価格変更により {abs(int(total_difference)):,}円 の売上減少があったと推定されます。価格戦略の見直しが必要かもしれません。\n")
                else:
                    f.write(f"全体分析: 期間全体で価格変更による売上への顕著な影響は見られませんでした。\n")
    except Exception as e: st.warning(f"サマリーテキスト保存エラー: {e}")
    
    try:
        error_data = [meta for meta in metadata_list if not meta.get("success", False)] if metadata_list else []
        if error_data:
            error_df = pd.DataFrame([
                {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
                for meta in error_data
            ])
            error_df.to_csv(os.path.join(results_dir, "error_details.csv"), index=False, encoding='utf-8-sig')
    except Exception as e: st.warning(f"エラー詳細CSV保存エラー: {e}")
    
    st.success(f"🗂️ 結果の保存処理が完了しました。保存先: {os.path.abspath(results_dir)}")
    return os.path.abspath(results_dir)

# この関数をページ内で結果表示用に使用する
def display_batch_results_in_page(metadata_list: Optional[List[Dict[str, Any]]], return_figures: bool = False) -> Optional[Tuple[Optional[pd.DataFrame], Optional[Any], Optional[Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """バッチ処理結果の集計表示（ページ内での表示用）"""
    if not metadata_list:
        st.warning("バッチ処理結果がありません")
        return (None, None, None, None, None) if return_figures else None
        
    success_count = sum(1 for meta in metadata_list if meta.get("success", False))
    fail_count = len(metadata_list) - success_count
    
    st.metric("処理総数", f"{len(metadata_list)}件", f"成功: {success_count}件, 失敗: {fail_count}件")
    
    result_df_display: Optional[pd.DataFrame] = None
    fig_date_display: Optional[Any] = None
    fig_class_display: Optional[Any] = None
    date_revenue_df_display: Optional[pd.DataFrame] = None
    class_revenue_df_display: Optional[pd.DataFrame] = None

    if success_count == 0:
        st.error("すべての処理が失敗しました")
        error_df_display = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
            for meta in metadata_list if not meta.get("success", False)
        ])
        st.dataframe(error_df_display)
        if return_figures:
            return error_df_display, None, None, None, None
        return None
    
    success_data = [meta for meta in metadata_list if meta.get("success", False)]
    total_actual = sum(meta.get("revenue_actual", 0) for meta in success_data)
    total_predicted = sum(meta.get("revenue_predicted", 0) for meta in success_data)
    total_difference = sum(meta.get("revenue_difference", 0) for meta in success_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("実績売上", f"{int(total_actual):,}円")
    with col2:
        st.metric("予測売上（価格変更影響なし時は実績値）", f"{int(total_predicted):,}円")
    with col3:
        delta_color = "normal" if total_difference >= 0 else "inverse"
        st.metric("売上差額（実績-予測）", f"{int(total_difference):,}円", 
                delta=f"{int(total_difference):,}円", delta_color=delta_color)
    
    result_data_for_df = []
    for meta in success_data:
        result_entry = {
             "利用日": meta.get("date"),
             "車両クラス": meta.get("car_class"),
             "使用モデル": meta.get("model_name", "不明"),
             "価格変更リードタイム": meta.get("last_change_lt"),
             "利用台数累積実績（価格変更時）": meta.get("actual_usage_at_change_lt", np.nan),
             "利用台数累積実績": meta.get("actual_cumulative_usage_end", np.nan),
             "利用台数累積（予測）": meta.get("predicted_cumulative_usage_end", np.nan),
             "追加実績予約数（価格変更後）": meta.get("additional_actual_bookings", np.nan),
             "追加予測予約数（価格変更後）": meta.get("additional_predicted_bookings", np.nan),
             "変更前価格": meta.get("price_before_change", np.nan),
             "変更後価格": meta.get("price_after_change", np.nan),
             "実績売上（変更前）": meta.get("revenue_actual_before", np.nan),
             "実績売上（変更後）": meta.get("revenue_actual_after", np.nan),
             "予測売上（変更後）": meta.get("revenue_predicted_after", np.nan),
             "実績売上": meta.get("revenue_actual", 0),
             "予測売上（価格変更影響なし時は実績値）": meta.get("revenue_predicted", 0),
             "売上差額": meta.get("revenue_difference", 0)
         }
        result_data_for_df.append(result_entry)

    result_df_display = pd.DataFrame(result_data_for_df)
    result_df_numeric = result_df_display.copy()
    
    st.subheader("詳細結果")
    display_columns_order = [
        "利用日", "車両クラス", "使用モデル", "価格変更リードタイム",
        "利用台数累積実績（価格変更時）",
        "利用台数累積実績",
        "利用台数累積（予測）",
        "追加実績予約数（価格変更後）",
        "追加予測予約数（価格変更後）",
        "変更前価格", "変更後価格",
        "実績売上（変更前）",
        "実績売上（変更後）",
        "予測売上（変更後）",
        "実績売上",
        "予測売上（価格変更影響なし時は実績値）",
        "売上差額"
    ]
    cols_to_display = [col for col in result_df_numeric.columns if col in display_columns_order]

    st.subheader("日付別売上差額")
    date_revenue_df_display = result_df_numeric.groupby("利用日").agg({
        "実績売上": "sum", "予測売上（価格変更影響なし時は実績値）": "sum", "売上差額": "sum"
    }).reset_index()
    fig_date_display = None
    if not date_revenue_df_display.empty:
        df_for_plot_date = date_revenue_df_display.rename(columns={"予測売上（価格変更影響なし時は実績値）": "予測売上"})
        fig_date_display = plot_batch_revenue_comparison(df_for_plot_date, "利用日")
        st.plotly_chart(fig_date_display, use_container_width=True, key="date_revenue_chart")
    else:
        st.info("日付別売上データがありません。")
    
    st.subheader("車両クラス別売上差額")
    class_revenue_df_display = result_df_numeric.groupby("車両クラス").agg({
        "実績売上": "sum", "予測売上（価格変更影響なし時は実績値）": "sum", "売上差額": "sum"
    }).reset_index()
    fig_class_display = None
    if not class_revenue_df_display.empty:
        df_for_plot_class = class_revenue_df_display.rename(columns={"予測売上（価格変更影響なし時は実績値）": "予測売上"})
        fig_class_display = plot_batch_revenue_comparison(df_for_plot_class, "車両クラス", horizontal=True)
        st.plotly_chart(fig_class_display, use_container_width=True, key="class_revenue_chart")
    else:
        st.info("車両クラス別売上データがありません。")
        
    st.subheader("価格変更幅 vs 追加予約数インパクト")
    fig_impact = plot_price_change_vs_booking_impact(result_df_numeric)
    if fig_impact.data:
        st.plotly_chart(fig_impact, use_container_width=True, key="price_booking_impact_chart")
    else:
        pass

    st.subheader("詳細結果テーブル")

    column_config_dict = {
        "利用台数累積実績": st.column_config.NumberColumn(format="%.1f"),
        "利用台数累積（予測）": st.column_config.NumberColumn(format="%.1f"),
        "利用台数累積実績（価格変更時）": st.column_config.NumberColumn(format="%.1f"),
        "追加実績予約数（価格変更後）": st.column_config.NumberColumn(format="%.1f"),
        "追加予測予約数（価格変更後）": st.column_config.NumberColumn(format="%.1f"),
        "変更前価格": st.column_config.NumberColumn(format="%.1f"),
        "変更後価格": st.column_config.NumberColumn(format="%.1f"),
    }
    st.dataframe(
        result_df_numeric[cols_to_display].sort_values(by=["利用日", "車両クラス"]),
        column_config=column_config_dict,
    )

    # --- ダウンロードボタン (CSV出力時にフォーマット指定) ---
    csv_download = result_df_numeric.to_csv(
        index=False,
        float_format='%.1f' # Format all floats to 1 decimal place for CSV
    ).encode('utf-8')
    download_filename = f"batch_analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.download_button("💾 集計結果をダウンロード (数値)", csv_download, download_filename, "text/csv", key="download_batch_csv_page")
    
    if total_difference > 0:
        st.success(f"**全体分析**: 期間全体で価格変更により **{int(total_difference):,}円** の追加売上が発生したと推定されます。価格戦略は有効に機能しています。")
    elif total_difference < 0:
        st.warning(f"**全体分析**: 期間全体で価格変更により **{abs(int(total_difference)):,}円** の売上減少があったと推定されます。価格戦略の見直しが必要かもしれません。")
    else:
        st.info("**全体分析**: 期間全体で価格変更による売上への顕著な影響は見られませんでした。")
        
    if fail_count > 0:
        st.markdown("---")
        st.subheader("失敗詳細")
        error_df_page = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
            for meta in metadata_list if not meta.get("success", False)
        ])
        st.dataframe(error_df_page)
    
    if return_figures:
        return result_df_numeric, fig_date_display, fig_class_display, date_revenue_df_display, class_revenue_df_display
    return None

def render_batch_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    """複数日付・車両クラスのバッチ分析ページを描画"""
    st.title("複数日付範囲での集計分析")
    
    _, _, _ = render_prediction_sidebar_widgets(data)
    saved_models = list_saved_models()
    if not saved_models:
        st.warning("予測を実行するには、まず「モデルトレーニング」ページでモデルを作成してください。")
        return
    
    st.header("分析対象設定")
    st.markdown("複数の日付と車両クラスの組み合わせに対して一括で予測分析を行います。")
    with st.expander("分析手法の説明", expanded=True):
        st.markdown("""
        ### 売上金額影響分析の仕組み
        この分析ツールは、**価格変更が売上に与えた影響**を定量的に評価します。
        1. **価格変更点の特定**: 各日付・車両クラスごとに価格変更があった最後のリードタイム（LT）を特定。
        2. **実績売上の計算**: 実際の予約データに基づく売上。
        3. **予測売上の計算**: 「価格が変更されなかった場合」のシナリオに基づく予測売上。
        4. **売上差額の算出**: 実績売上 - 予測売上 = 価格変更による影響額。
        """)
    
    date_range = []
    with st.expander("日付範囲の選択", expanded=True):
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]):
            available_dates = sorted(data[DATE_COLUMN].dt.date.unique())
            if available_dates:
                date_selection_method = st.radio("日付選択方法", ["日付範囲を指定", "個別の日付を選択"], horizontal=True, key="batch_date_select_method")
                if date_selection_method == "日付範囲を指定":
                    col1, col2 = st.columns(2)
                    min_date_val, max_date_val = available_dates[0], available_dates[-1]
                    default_start = datetime.date(2025, 4, 1)
                    default_start = default_start if min_date_val <= default_start <= max_date_val else min_date_val
                    start_date = st.date_input("開始日", default_start, min_value=min_date_val, max_value=max_date_val, key="batch_start_date")
                    default_end = datetime.date(2025, 4, 14)
                    default_end = default_end if start_date <= default_end <= max_date_val else max_date_val
                    end_date = st.date_input("終了日", default_end, min_value=start_date, max_value=max_date_val, key="batch_end_date")
                    date_range = [d for d in available_dates if start_date <= d <= end_date]
                else:
                    date_options = [d.strftime('%Y-%m-%d') for d in available_dates]
                    default_selection = [d.strftime('%Y-%m-%d') for d in available_dates if datetime.date(2025, 4, 1) <= d <= datetime.date(2025, 4, 14)]
                    if not default_selection: default_selection = date_options[:min(5, len(date_options))]
                    selected_dates_str = st.multiselect("分析する日付を選択", date_options, default=default_selection, key="batch_multi_date")
                    date_range = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in selected_dates_str]
                st.info(f"選択された日付: {len(date_range)}日")
                if date_range: st.dataframe(pd.DataFrame([{'利用日': d} for d in date_range]))
                else: st.warning("日付を選択してください")
            else: st.warning("利用可能な日付がありません")
        else: st.error(f"'{DATE_COLUMN}'列がないか日付型ではありません")
    
    car_classes_selected_ui = []
    models_for_run = {}
    with st.expander("車両クラスとモデルの選択", expanded=True):
        if CAR_CLASS_COLUMN in data.columns:
            all_car_classes = sorted(data[CAR_CLASS_COLUMN].unique())
            models_by_class = {cls: [m for m in saved_models if m.get("car_class") == cls or m.get("car_class") == "全クラス"] for cls in all_car_classes}
            class_sel_method = st.radio("車両クラス選択方法", ["すべての車両クラス", "個別の車両クラスを選択"], horizontal=True, key="batch_class_select_method")
            if class_sel_method == "すべての車両クラス": car_classes_selected_ui = all_car_classes
            else: car_classes_selected_ui = st.multiselect("分析する車両クラスを選択", all_car_classes, default=[all_car_classes[0]] if all_car_classes else [], key="batch_multi_class")
            
            if car_classes_selected_ui:
                st.info(f"選択された車両クラス: {len(car_classes_selected_ui)}クラス")
                st.subheader("車両クラスごとのモデル選択")
                for cls_item in car_classes_selected_ui:
                    mods_for_cls = models_by_class.get(cls_item, [])
                    if not mods_for_cls: st.warning(f"'{cls_item}'に対応するモデルなし"); continue
                    mod_opts = [f"{m['model_name']} ({m['model_type']})" for m in mods_for_cls]
                    def_idx = next((i for i, m in enumerate(mods_for_cls) if m.get("car_class") == cls_item), 0)
                    c1, c2 = st.columns([2,3])
                    with c1: st.markdown(f"**{cls_item}**")
                    with c2:
                        sel_mod_idx = st.selectbox(f"モデルを選択", range(len(mod_opts)), format_func=lambda i: mod_opts[i], index=def_idx, key=f"sel_mod_{cls_item}")
                        models_for_run[cls_item] = mods_for_cls[sel_mod_idx]
                        if "metrics" in models_for_run[cls_item]: 
                            metrics_disp = models_for_run[cls_item]["metrics"]
                            metric_str_disp = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_disp.items() if k in ["RMSE", "R2"]])
                            st.caption(f"学習データ行数: {models_for_run[cls_item].get('row_count', '不明')}行, {metric_str_disp}")
                st.subheader("選択されたモデル一覧")
                display_model_data = []
                for cls, model_detail in models_for_run.items():
                    row = {"車両クラス": cls, "モデル名": model_detail.get("model_name", "N/A"), "モデルタイプ": model_detail.get("model_type", "N/A")}
                    metrics = model_detail.get("metrics", {})
                    row["RMSE"] = metrics.get("RMSE", "N/A")
                    row["R2"] = metrics.get("R2", "N/A")
                    row["学習データ行数"] = model_detail.get("row_count", "N/A")
                    display_model_data.append(row)
                if display_model_data: st.dataframe(pd.DataFrame(display_model_data))

            else: st.warning("車両クラスを選択してください")
        else: st.error(f"'{CAR_CLASS_COLUMN}'列なし")

    with st.expander("実行設定", expanded=True):
        max_workers_val = st.slider("並列処理数", 1, (os.cpu_count() or 1), min(4, (os.cpu_count() or 1)), key="batch_max_workers")
        combinations = len(date_range) * len(models_for_run)
        st.info(f"処理予定: {combinations}件 (日付{len(date_range)}×クラス{len(models_for_run)})")
        estimated_time_sec = combinations * 1.5 / max_workers_val
        st.warning(f"処理時間目安: 約 {estimated_time_sec:.1f}秒（{estimated_time_sec/60:.1f}分）")
        save_locally_flag = st.checkbox("結果をローカルフォルダに保存", True, key="batch_save_local")
    
    if date_range and models_for_run and st.button("🚀 バッチ分析を実行", key="batch_run_main_btn"):
        st.markdown("---"); st.header("バッチ分析結果")
        if date_range: 
            min_display_date = min(date_range)
            max_display_date = max(date_range)
            st.subheader(f"分析期間: {min_display_date}{' 〜 ' + str(max_display_date) if min_display_date != max_display_date else ''}")
        
        with st.spinner('バッチ処理実行中...'):
            loaded_models_dict = {}
            loaded_metadata_dict = {}
            for cls_run, model_info_run in models_for_run.items():
                lm = load_model(model_info_run["path"])
                if lm: loaded_models_dict[cls_run] = lm
                else: st.error(f"{cls_run}のモデルロード失敗"); continue 
                if "filename" in model_info_run: loaded_metadata_dict[cls_run] = get_model_metadata(model_info_run["filename"])
            
            if not loaded_models_dict: st.error("実行可能モデルがありませんでした。"); return

            actual_car_classes_to_run = list(loaded_models_dict.keys())
            _, result_metadata_list = run_batch_prediction(
                data, loaded_models_dict, date_range, actual_car_classes_to_run, loaded_metadata_dict, max_workers_val
            )
            
            display_data_tuple = display_batch_results_in_page(result_metadata_list, return_figures=True)
            
            if save_locally_flag and display_data_tuple:
                df_res, fig_d_res, fig_c_res, df_date_rev_res, df_class_rev_res = display_data_tuple
                if df_res is not None: 
                    save_batch_results_to_folder(
                        result_metadata_list, df_date_rev_res, df_class_rev_res, df_res, fig_d_res, fig_c_res
                    )
    st.markdown("---"); st.subheader("データプレビュー"); st.dataframe(data.head()) 