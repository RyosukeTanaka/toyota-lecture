# utils/page_batch_analysis.py

import streamlit as st
import pandas as pd
import datetime
import os
from typing import Dict, Any, List
from .constants import (
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN
)
from .ui_components import render_prediction_sidebar_widgets
from .model_storage import load_model, get_model_metadata, list_saved_models
from .batch_analysis import run_batch_prediction, display_batch_results
from .visualization import plot_batch_revenue_comparison


def save_batch_results_to_folder(
    metadata_list: List[Dict[str, Any]], 
    date_revenue_df: pd.DataFrame, 
    class_revenue_df: pd.DataFrame,
    result_df: pd.DataFrame,
    fig_date: Any,
    fig_class: Any
) -> str:
    """バッチ分析結果をローカルフォルダに保存する

    Parameters
    ----------
    metadata_list : List[Dict[str, Any]]
        バッチ予測のメタデータリスト
    date_revenue_df : pd.DataFrame
        日付別売上差額データフレーム
    class_revenue_df : pd.DataFrame
        車両クラス別売上差額データフレーム
    result_df : pd.DataFrame
        詳細結果データフレーム
    fig_date : Any
        日付別売上差額グラフ
    fig_class : Any
        車両クラス別売上差額グラフ

    Returns
    -------
    str
        保存先フォルダのパス
    """
    # 結果保存用のフォルダを作成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"batch_results_{timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # CSVデータの保存
    result_df.to_csv(f"{results_dir}/batch_analysis_results.csv", index=False, encoding='utf-8-sig')
    date_revenue_df.to_csv(f"{results_dir}/date_revenue_summary.csv", index=False, encoding='utf-8-sig')
    class_revenue_df.to_csv(f"{results_dir}/class_revenue_summary.csv", index=False, encoding='utf-8-sig')
    
    # グラフの保存
    fig_date.write_image(f"{results_dir}/date_revenue_chart.png")
    fig_class.write_image(f"{results_dir}/class_revenue_chart.png")
    fig_date.write_html(f"{results_dir}/date_revenue_chart.html")
    fig_class.write_html(f"{results_dir}/class_revenue_chart.html")
    
    # サマリーテキストの保存
    success_count = sum(1 for meta in metadata_list if meta.get("success", False))
    fail_count = len(metadata_list) - success_count
    
    # 成功したデータの集計
    success_data = [meta for meta in metadata_list if meta.get("success", False)]
    
    # 売上差額の合計
    total_actual = sum(meta.get("revenue_actual", 0) for meta in success_data)
    total_predicted = sum(meta.get("revenue_predicted", 0) for meta in success_data)
    total_difference = sum(meta.get("revenue_difference", 0) for meta in success_data)
    
    with open(f"{results_dir}/summary.txt", "w", encoding='utf-8') as f:
        f.write(f"バッチ分析サマリー\n")
        f.write(f"実行日時: {timestamp}\n")
        f.write(f"=================================\n")
        f.write(f"処理総数: {len(metadata_list)}件\n")
        f.write(f"成功: {success_count}件, 失敗: {fail_count}件\n\n")
        f.write(f"売上集計結果:\n")
        f.write(f"実績総売上: {int(total_actual):,}円\n")
        f.write(f"予測総売上（価格固定）: {int(total_predicted):,}円\n")
        f.write(f"売上差額（実績-予測）: {int(total_difference):,}円\n\n")
        
        if total_difference > 0:
            f.write(f"全体分析: 期間全体で価格変更により {int(total_difference):,}円 の追加売上が発生したと推定されます。価格戦略は有効に機能しています。\n")
        elif total_difference < 0:
            f.write(f"全体分析: 期間全体で価格変更により {abs(int(total_difference)):,}円 の売上減少があったと推定されます。価格戦略の見直しが必要かもしれません。\n")
        else:
            f.write(f"全体分析: 期間全体で価格変更による売上への顕著な影響は見られませんでした。\n")
    
    # 失敗詳細の保存
    error_data = [meta for meta in metadata_list if not meta.get("success", False)]
    if error_data:
        error_df = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
            for meta in error_data
        ])
        error_df.to_csv(f"{results_dir}/error_details.csv", index=False, encoding='utf-8-sig')
    
    st.success(f"🗂️ 分析結果をフォルダに保存しました: {os.path.abspath(results_dir)}")
    return os.path.abspath(results_dir)


def render_batch_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    """複数日付・車両クラスのバッチ分析ページを描画

    Parameters
    ----------
    data : pd.DataFrame
        分析用データフレーム
    config : Dict[str, Any]
        設定情報辞書
    """
    st.title("複数日付範囲での集計分析")
    
    # サイドバーウィジェットの描画
    # 注：ここではサイドバーのモデル選択結果は使用せず、メインエリアで改めてモデル選択UIを提供
    # ただし、初期ロード時のエラーを避けるためにサイドバーウィジェットは維持
    (
        _,  # selected_car_class (使用しない)
        _,  # selected_model_info (使用しない)
        _   # 予測実行ボタン (使用しない)
    ) = render_prediction_sidebar_widgets(data)
    
    # --- メインエリア --- #
    # 保存済みモデルを取得
    saved_models = list_saved_models()
    
    if not saved_models:
        st.warning("予測を実行するには、まず「モデルトレーニング」ページでモデルを作成してください。")
        return
    
    # --- バッチ分析の入力パラメータ設定 --- #
    st.header("分析対象設定")
    st.markdown("複数の日付と車両クラスの組み合わせに対して一括で予測分析を行います。")
    
    # 分析手法の説明を追加
    with st.expander("分析手法の説明", expanded=True):
        st.markdown("""
        ### 売上金額影響分析の仕組み
        
        この分析ツールは、**価格変更が売上に与えた影響**を定量的に評価します。計算方法は以下の通りです：
        
        1. **価格変更点の特定**：
           - 各日付・車両クラスごとに価格変更があった最後のリードタイム（LT）を特定
           - このポイントを基準点として分析
        
        2. **実績売上の計算**：
           - 実際に発生した予約データに基づいた売上
           - 各リードタイムでの日次新規予約数 × その時点の実際価格
        
        3. **予測売上の計算**：
           - 「価格が変更されなかった場合」のシナリオに基づく予測売上
           - 機械学習モデルが予測した日次新規予約数 × 価格変更点の固定価格
        
        4. **売上差額の算出**：
           - 実績売上 - 予測売上 = 価格変更による影響額
           - プラスの値：価格戦略により売上増加
           - マイナスの値：価格戦略により売上減少
        
        この分析により、価格変更の効果を数値で確認できます。例えば価格を上げたことで予約数が減っても総売上が増加したケースや、価格を下げて予約数を増やした効果を確認できます。
        """)
    
    # 日付範囲選択
    date_range = []
    with st.expander("日付範囲の選択", expanded=True):
        # 有効な日付リストを取得
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]):
            available_dates = sorted(data[DATE_COLUMN].dt.date.unique())
            
            if len(available_dates) > 0:
                # 日付選択方法
                date_selection_method = st.radio(
                    "日付選択方法",
                    ["日付範囲を指定", "個別の日付を選択"],
                    horizontal=True
                )
                
                if date_selection_method == "日付範囲を指定":
                    col1, col2 = st.columns(2)
                    with col1:
                        min_date = available_dates[0]
                        max_date = available_dates[-1]
                        # デフォルト値を2025/04/01に設定
                        default_start_date = datetime.date(2025, 4, 1)
                        # データの範囲外の場合はデータ範囲内の最初の日付を使用
                        if default_start_date < min_date or default_start_date > max_date:
                            default_start_date = min_date
                            st.info(f"指定されたデフォルト開始日 2025/04/01 がデータ範囲外のため、最初の利用日 {min_date} を使用します。")
                        start_date = st.date_input("開始日", default_start_date, min_value=min_date, max_value=max_date)
                    with col2:
                        # デフォルト値を2025/04/14に設定
                        default_end_date = datetime.date(2025, 4, 14)
                        # データの範囲外またはstart_dateより前の場合はデータ範囲内の最後の日付を使用
                        if default_end_date > max_date or default_end_date < start_date:
                            default_end_date = max_date
                            st.info(f"指定されたデフォルト終了日 2025/04/14 がデータ範囲外のため、最後の利用日 {max_date} を使用します。")
                        end_date = st.date_input("終了日", default_end_date, min_value=start_date, max_value=max_date)
                    
                    # 開始日と終了日の間の日付をすべて取得
                    date_range = [d for d in available_dates if start_date <= d <= end_date]
                    st.info(f"選択された日付範囲に含まれる利用日: {len(date_range)}日")
                    
                else:  # 個別の日付を選択
                    date_options = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in available_dates]
                    
                    # 2025/04/01から2025/04/14の範囲の日付をデフォルトで選択
                    default_start = datetime.date(2025, 4, 1)
                    default_end = datetime.date(2025, 4, 14)
                    
                    # デフォルト選択する日付リストを作成
                    default_dates = []
                    for d in available_dates:
                        if default_start <= d <= default_end:
                            default_dates.append(datetime.datetime.strftime(d, '%Y-%m-%d'))
                    
                    # デフォルト日付が範囲外の場合のフォールバック
                    if not default_dates:
                        default_dates = date_options[:min(5, len(date_options))]  # 最初の5つをデフォルトに
                        st.info(f"指定された日付範囲（2025/04/01～2025/04/14）がデータに含まれていないため、最初の{len(default_dates)}日を選択します。")
                    else:
                        st.info(f"デフォルトで2025/04/01～2025/04/14の範囲にある{len(default_dates)}日を選択しています。")
                    
                    selected_dates = st.multiselect(
                        "分析する日付を選択",
                        options=date_options,
                        default=default_dates
                    )
                    date_range = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in selected_dates]
                    st.info(f"選択された日付: {len(date_range)}日")
                
                # 選択された日付を表示
                if date_range:
                    st.subheader("選択された日付一覧")
                    date_df = pd.DataFrame([{'利用日': d} for d in date_range])
                    st.dataframe(date_df)
                else:
                    st.warning("少なくとも1つの日付を選択してください")
            else:
                st.warning("利用可能な日付がありません")
        else:
            st.error(f"'{DATE_COLUMN}'列がないか日付型ではありません")
    
    # 車両クラス選択とモデル割り当て
    car_classes = []
    selected_models = {}  # 車両クラスごとに選択されたモデル情報を格納する辞書
    
    with st.expander("車両クラスとモデルの選択", expanded=True):
        if CAR_CLASS_COLUMN in data.columns:
            available_classes = sorted(data[CAR_CLASS_COLUMN].unique())
            
            # 保存済みモデルを車両クラスごとにグループ化
            models_by_class = {}
            for car_class in available_classes:
                # 特定の車両クラス専用モデルと全クラス対応モデルをリストアップ
                models_by_class[car_class] = [
                    model for model in saved_models 
                    if model.get("car_class") == car_class or model.get("car_class") == "全クラス"
                ]
            
            # 車両クラス選択方法
            class_selection_method = st.radio(
                "車両クラス選択方法",
                ["すべての車両クラス", "個別の車両クラスを選択"],
                horizontal=True
            )
            
            if class_selection_method == "すべての車両クラス":
                car_classes = available_classes
                st.info(f"すべての車両クラス: {len(car_classes)}クラス")
            else:  # 個別の車両クラスを選択
                selected_classes = st.multiselect(
                    "分析する車両クラスを選択",
                    options=available_classes,
                    default=[available_classes[0]] if available_classes else []
                )
                car_classes = selected_classes
                st.info(f"選択された車両クラス: {len(car_classes)}クラス")
            
            # 選択された車両クラスを表示
            if car_classes:
                st.subheader("車両クラスごとのモデル選択")
                
                # 各車両クラスに対してモデル選択UIを表示
                for car_class in car_classes:
                    # クラス専用モデルと全クラス対応モデルをフィルタリング
                    available_models = models_by_class.get(car_class, [])
                    
                    if not available_models:
                        st.warning(f"'{car_class}'に対応するモデルが見つかりません。")
                        continue
                    
                    # モデル名のリストを作成（表示用）
                    model_options = [f"{model['model_name']} ({model['model_type']})" for model in available_models]
                    
                    # デフォルト選択（車両クラス専用モデルを優先）
                    default_index = 0
                    for i, model in enumerate(available_models):
                        if model.get("car_class") == car_class:
                            default_index = i
                            break
                    
                    # モデル選択
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(f"**{car_class}**")
                    with col2:
                        selected_model_index = st.selectbox(
                            f"モデルを選択",
                            options=range(len(model_options)),
                            format_func=lambda i: model_options[i],
                            index=default_index,
                            key=f"model_select_{car_class}"
                        )
                        
                        # 選択されたモデル情報を保存
                        selected_model = available_models[selected_model_index]
                        selected_models[car_class] = selected_model
                        
                        # モデル情報表示（小さく）
                        if "metrics" in selected_model:
                            metrics = selected_model["metrics"]
                            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k in ["RMSE", "R2"]])
                            st.caption(f"学習データ行数: {selected_model.get('row_count', '不明')}行, {metric_str}")
                
                # 選択されたモデル一覧をテーブルで表示
                st.subheader("選択されたモデル一覧")
                model_selection_data = []
                for car_class, model in selected_models.items():
                    model_selection_data.append({
                        "車両クラス": car_class,
                        "モデル名": model["model_name"],
                        "モデルタイプ": model["model_type"],
                        "RMSE": model.get("metrics", {}).get("RMSE", "N/A"),
                        "R2": model.get("metrics", {}).get("R2", "N/A"),
                        "学習データ行数": model.get("row_count", "不明")
                    })
                model_selection_df = pd.DataFrame(model_selection_data)
                st.dataframe(model_selection_df)
            else:
                st.warning("少なくとも1つの車両クラスを選択してください")
        else:
            st.error(f"'{CAR_CLASS_COLUMN}'列が存在しません")
    
    # 実行設定
    with st.expander("実行設定", expanded=True):
        st.markdown("#### 並列処理の設定")
        max_workers = st.slider("並列処理数", min_value=1, max_value=8, value=4, 
                              help="同時に処理するタスクの数を指定します。大きすぎると処理が不安定になる可能性があります。")
        
        total_combinations = len(date_range) * len(car_classes)
        estimated_time = total_combinations * 1.5 / max_workers  # 1.5秒 × 組み合わせ数 / 並列数
        
        st.info(f"処理予定の組み合わせ: 日付 {len(date_range)}個 × 車両クラス {len(car_classes)}個 = 合計 {total_combinations}件")
        st.warning(f"処理時間目安: 約 {estimated_time:.1f}秒（{estimated_time/60:.1f}分）")
        
        # 結果保存チェックボックス
        save_results = st.checkbox("結果をローカルフォルダに保存する", value=True, 
                                help="チェックすると、分析結果のグラフ、CSVファイル、サマリーをローカルフォルダに保存します。")
    
    # 実行ボタン
    run_batch = False
    if date_range and car_classes and selected_models:
        run_batch = st.button("🚀 バッチ分析を実行", key="run_batch", 
                             help="選択された日付・車両クラスの組み合わせに対してバッチ処理を実行します")
    
    # バッチ処理を実行
    if run_batch:
        st.markdown("---")
        st.header("バッチ分析結果")
        
        # 選択された日付期間を表示
        if date_range:
            min_date = min(date_range)
            max_date = max(date_range)
            if min_date == max_date:
                st.subheader(f"分析期間: {min_date}")
            else:
                st.subheader(f"分析期間: {min_date} 〜 {max_date}")
        
        with st.spinner('バッチ処理を実行中...'):
            # 各モデルをロードして辞書に格納
            models_dict = {}
            models_metadata_dict = {}
            
            for car_class, model_info in selected_models.items():
                # モデルのロード
                model = load_model(model_info["path"])
                
                if model is None:
                    st.error(f"{car_class}用のモデル '{model_info['model_name']}' の読み込みに失敗しました。")
                    continue
                
                # モデルのメタデータを取得
                model_metadata = None
                if "filename" in model_info:
                    model_metadata = get_model_metadata(model_info["filename"])
                
                # 辞書に格納
                models_dict[car_class] = model
                models_metadata_dict[car_class] = model_metadata
            
            if not models_dict:
                st.error("モデルの読み込みに失敗しました。")
                return
            
            # バッチ予測実行（車両クラスごとにモデルを切り替える）
            predictions, metadata_list = run_batch_prediction(
                data=data,
                models=models_dict,  # モデル辞書を渡す
                date_range=date_range,
                car_classes=car_classes,
                models_metadata=models_metadata_dict,  # メタデータ辞書も渡す
                max_workers=max_workers
            )
            
            # 結果表示
            result_df, fig_date, fig_class, date_revenue_df, class_revenue_df = display_batch_results(metadata_list, return_figures=True)
            
            # 結果をローカルフォルダに保存
            if save_results and result_df is not None:
                saved_folder = save_batch_results_to_folder(
                    metadata_list, 
                    date_revenue_df, 
                    class_revenue_df,
                    result_df,
                    fig_date,
                    fig_class
                )
    
    # データプレビュー（常に表示しておく）
    st.markdown("---")
    st.subheader("データプレビュー")
    st.dataframe(data.head()) 


def display_batch_results(metadata_list: List[Dict[str, Any]], return_figures: bool = False):
    """バッチ処理結果の集計表示

    Parameters
    ----------
    metadata_list : List[Dict[str, Any]]
        バッチ予測のメタデータリスト
    return_figures : bool, default=False
        結果のグラフとデータフレームを返すかどうか

    Returns
    -------
    Tuple[pd.DataFrame, Any, Any, pd.DataFrame, pd.DataFrame] | None
        return_figures=Trueの場合、
        (結果データフレーム, 日付別グラフ, クラス別グラフ, 日付別データ, クラス別データ)
        を返す
    """
    if not metadata_list:
        st.warning("バッチ処理結果がありません")
        return None if not return_figures else (None, None, None, None, None)
        
    # 成功件数・失敗件数を集計
    success_count = sum(1 for meta in metadata_list if meta.get("success", False))
    fail_count = len(metadata_list) - success_count
    
    st.metric("処理総数", f"{len(metadata_list)}件", f"成功: {success_count}件, 失敗: {fail_count}件")
    
    # すべての処理が失敗した場合の処理
    if success_count == 0:
        st.error("すべての処理が失敗しました")
        # 失敗詳細はここで表示
        st.subheader("失敗詳細")
        error_df = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
            for meta in metadata_list if not meta.get("success", False)
        ])
        st.dataframe(error_df)
        return None if not return_figures else (error_df, None, None, None, None)
    
    # 成功したデータの集計
    success_data = [meta for meta in metadata_list if meta.get("success", False)]
    
    # 売上差額の合計
    total_actual = sum(meta.get("revenue_actual", 0) for meta in success_data)
    total_predicted = sum(meta.get("revenue_predicted", 0) for meta in success_data)
    total_difference = sum(meta.get("revenue_difference", 0) for meta in success_data)
    
    # 集計メトリクス表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("実績総売上", f"{int(total_actual):,}円")
    with col2:
        st.metric("予測総売上（価格固定）", f"{int(total_predicted):,}円")
    with col3:
        delta_color = "normal" if total_difference >= 0 else "inverse"
        st.metric("売上差額（実績-予測）", f"{int(total_difference):,}円", 
                delta=f"{int(total_difference):,}円", delta_color=delta_color)
    
    # 詳細データフレーム作成
    result_df = pd.DataFrame([
        {
            "利用日": meta.get("date"),
            "車両クラス": meta.get("car_class"),
            "使用モデル": meta.get("model_name", "不明"),
            "価格変更リードタイム": meta.get("last_change_lt"),
            "実績売上": int(meta.get("revenue_actual", 0)),
            "予測売上": int(meta.get("revenue_predicted", 0)),
            "売上差額": int(meta.get("revenue_difference", 0))
        }
        for meta in success_data
    ])
    
    # 詳細データ表示（ソートあり）
    st.subheader("詳細結果")
    st.dataframe(result_df.sort_values(by=["利用日", "車両クラス"]))
    
    # グラフ表示（日付ごとの集計）
    st.subheader("日付別売上差額")
    date_revenue_df = result_df.groupby("利用日").agg({
        "実績売上": "sum",
        "予測売上": "sum",
        "売上差額": "sum"
    }).reset_index()
    
    fig_date = plot_batch_revenue_comparison(date_revenue_df, "利用日")
    st.plotly_chart(fig_date, use_container_width=True)
    
    # グラフ表示（車両クラスごとの集計）
    st.subheader("車両クラス別売上差額")
    class_revenue_df = result_df.groupby("車両クラス").agg({
        "実績売上": "sum",
        "予測売上": "sum",
        "売上差額": "sum"
    }).reset_index()
    
    fig_class = plot_batch_revenue_comparison(class_revenue_df, "車両クラス", horizontal=True)
    st.plotly_chart(fig_class, use_container_width=True)
    
    # CSVダウンロード機能
    csv = result_df.to_csv(index=False).encode('utf-8')
    filename = f"batch_analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.download_button("💾 集計結果をダウンロード", csv, filename, "text/csv")
    
    # 要約テキスト
    if total_difference > 0:
        st.success(f"**全体分析**: 期間全体で価格変更により **{int(total_difference):,}円** の追加売上が発生したと推定されます。価格戦略は有効に機能しています。")
    elif total_difference < 0:
        st.warning(f"**全体分析**: 期間全体で価格変更により **{abs(int(total_difference)):,}円** の売上減少があったと推定されます。価格戦略の見直しが必要かもしれません。")
    else:
        st.info("**全体分析**: 期間全体で価格変更による売上への顕著な影響は見られませんでした。")
        
    # 失敗詳細をページの最下部に表示
    if fail_count > 0:
        st.markdown("---")
        st.subheader("失敗詳細")
        error_df = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "モデル": meta.get("model_name", "不明"), "エラー内容": meta.get("error", "不明")}
            for meta in metadata_list if not meta.get("success", False)
        ])
        st.dataframe(error_df)
    
    if return_figures:
        return result_df, fig_date, fig_class, date_revenue_df, class_revenue_df
    return None 