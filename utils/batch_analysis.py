# utils/batch_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import datetime
import concurrent.futures
from .constants import (
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN
)
from .data_processing import (
    filter_data_by_date, create_scenario_data,
    find_last_price_change_lead_time
)
from .model_storage import (
    load_model, prepare_features_for_prediction
)
from .revenue_analysis import calculate_revenue_difference
from .visualization import plot_batch_revenue_comparison


def batch_predict_date(
    data: pd.DataFrame,
    model,
    date: datetime.date,
    car_class: str,
    model_metadata: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """単一の日付とクラスの組み合わせに対して予測を実行する

    Parameters
    ----------
    data : pd.DataFrame
        元データフレーム
    model
        予測モデル
    date : datetime.date
        予測する利用日
    car_class : str
        車両クラス
    model_metadata : Optional[Dict], default=None
        モデルのメタデータ（特徴量情報）

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (予測結果データフレーム, 予測メタデータ)
    """
    result_meta = {
        "date": date,
        "car_class": car_class,
        "success": False,
        "error": None,
        "last_change_lt": None,
        "revenue_actual": 0,
        "revenue_predicted": 0,
        "revenue_difference": 0
    }
    
    try:
        # 指定された日付・車両クラスでデータをフィルタリング
        data_filtered = filter_data_by_date(
            data[data[CAR_CLASS_COLUMN] == car_class] if car_class != "全クラス" else data,
            DATE_COLUMN, date
        )
        
        if data_filtered.empty:
            result_meta["error"] = f"{date}の{car_class}データが存在しません"
            return pd.DataFrame(), result_meta
            
        if LEAD_TIME_COLUMN not in data_filtered.columns:
            result_meta["error"] = f"{LEAD_TIME_COLUMN}列が存在しません"
            return pd.DataFrame(), result_meta
            
        # データをリードタイムでソート
        data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)
        
        # 価格変更点を検知
        last_change_lt = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)
        result_meta["last_change_lt"] = last_change_lt
        
        if last_change_lt is None:
            result_meta["error"] = "価格変更点が見つかりません"
            return pd.DataFrame(), result_meta
            
        # シナリオデータを作成（価格変更点固定）
        data_scenario = create_scenario_data(
            data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN,
            scenario_type='last_change_fixed', change_lead_time=last_change_lt
        )
        
        if data_scenario.empty:
            result_meta["error"] = "シナリオデータの作成に失敗しました"
            return pd.DataFrame(), result_meta
            
        # データ前処理
        scen_data_transformed = data_scenario.copy()
        date_cols = scen_data_transformed.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            scen_data_transformed[col] = scen_data_transformed[col].dt.strftime('%Y-%m-%d')
        
        # カテゴリ変数の処理
        object_cols = scen_data_transformed.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col in model_metadata.get("model_columns", []):
                # カテゴリ変数をワンホットエンコーディングする必要がある場合は
                # prepare_features_for_predictionに任せる
                continue
            else:
                # モデルに直接渡す場合はobject型の列を削除
                scen_data_transformed = scen_data_transformed.drop(columns=[col])
        
        # ターゲット変数を除去
        if TARGET_VARIABLE in scen_data_transformed.columns:
            X = scen_data_transformed.drop(columns=[TARGET_VARIABLE])
        else:
            X = scen_data_transformed
            
        # 予測実行
        y_pred = None
        
        # 1. 特徴量変換を適用して予測を試行（優先）
        try:
            if model_metadata and "model_columns" in model_metadata:
                st.info(f"{date} {car_class}の特徴量変換を適用...")
                try:
                    # モデルのカラム情報を取得
                    model_feature_names = model_metadata.get("model_columns", [])
                    st.info(f"モデルが期待する特徴量: {model_feature_names}")
                    
                    # 特徴量変換を試みる
                    transformed_data = prepare_features_for_prediction(X, model_metadata)
                    
                    # 特徴量の不一致チェック
                    missing_features = [col for col in model_feature_names if col not in transformed_data.columns]
                    if missing_features:
                        st.warning(f"変換後も足りない特徴量があります: {missing_features}")
                        # 足りない特徴量を0で埋める（応急処置）
                        for col in missing_features:
                            transformed_data[col] = 0
                    
                    # 余分な特徴量の削除
                    extra_features = [col for col in transformed_data.columns if col not in model_feature_names]
                    if extra_features:
                        st.warning(f"モデルに不要な特徴量を削除します: {extra_features}")
                        transformed_data = transformed_data.drop(columns=extra_features)
                    
                    # 特徴量の順序をモデルに合わせる
                    transformed_data = transformed_data[model_feature_names]
                    
                    # 最終チェック
                    if list(transformed_data.columns) != model_feature_names:
                        st.error(f"特徴量順序の不一致: {list(transformed_data.columns)} vs {model_feature_names}")
                    else:
                        st.success("特徴量の順序と数が一致しました！")
                    
                    # カテゴリ変数の追加チェック（念のため）
                    cat_cols = transformed_data.select_dtypes(include=['object']).columns
                    if not cat_cols.empty:
                        st.warning(f"変換後もobject型の列が残っています: {list(cat_cols)}。数値型に変換します。")
                        for col in cat_cols:
                            transformed_data[col] = pd.factorize(transformed_data[col])[0]
                    
                    y_pred = model.predict(transformed_data)
                    
                except Exception as transform_error:
                    st.error(f"特徴量変換中にエラー: {transform_error}")
                    raise transform_error  # 次の方法を試すためにエラーを再度発生させる
            else:
                # model_metadataがない場合、object型の列を削除して直接予測を試みる
                st.warning(f"{date} {car_class}のモデルメタデータがないため、直接予測を試みます。")
                numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
                X_numeric = X[numeric_cols]
                st.warning(f"数値型の列のみを使用します: {list(numeric_cols)}")
                y_pred = model.predict(X_numeric)
        except Exception as e1:
            # 2. 直接予測を試行（フォールバック）
            try:
                if hasattr(model, 'predict'):
                    st.info(f"{date} {car_class}のフォールバック予測を試行...")
                    
                    # 学習したモデルから特徴量情報を取得
                    model_features = []
                    if hasattr(model, 'feature_names_in_'):
                        model_features = list(model.feature_names_in_)
                        st.info(f"モデルから直接取得した特徴量: {model_features}")
                    
                    if model_features:
                        # モデルが使用する特徴量の情報がある場合は、それに合わせる
                        # 数値特徴量だけを選択
                        numeric_X = X.select_dtypes(include=['number', 'bool'])
                        available_features = [col for col in model_features if col in numeric_X.columns]
                        
                        if available_features:
                            # 利用可能な特徴量だけを選択
                            X_selected = numeric_X[available_features]
                            
                            # 足りない特徴量を0で埋める
                            missing_features = [col for col in model_features if col not in X_selected.columns]
                            for col in missing_features:
                                X_selected[col] = 0
                                
                            # 正しい順序に並べ替え
                            X_selected = X_selected[model_features]
                            st.warning(f"選択した特徴量: {X_selected.columns.tolist()}")
                            y_pred = model.predict(X_selected)
                        else:
                            # 利用可能な特徴量がない場合はデフォルトに戻る
                            numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
                            X_numeric = X[numeric_cols]
                            st.warning(f"数値型の列のみを使用します: {list(numeric_cols)}")
                            y_pred = model.predict(X_numeric)
                    else:
                        # モデル特徴量情報がない場合は数値型のみ
                        numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
                        X_numeric = X[numeric_cols]
                        st.warning(f"数値型の列のみを使用します: {list(numeric_cols)}")
                        y_pred = model.predict(X_numeric)
            except Exception as e2:
                result_meta["error"] = f"予測に失敗しました: {str(e2)}"
                return pd.DataFrame(), result_meta
                
        if y_pred is None:
            result_meta["error"] = "予測結果がNoneです"
            return pd.DataFrame(), result_meta
            
        # 予測結果をデータフレームに変換
        predictions_result = data_scenario.copy()
        predictions_result['prediction_label'] = y_pred
        
        # 売上差額の計算
        revenue_df, total_actual, total_predicted, total_difference = calculate_revenue_difference(
            df_actual=data_filtered_sorted,
            df_predicted=predictions_result,
            lead_time_col=LEAD_TIME_COLUMN,
            actual_usage_col=TARGET_VARIABLE,
            pred_usage_col='prediction_label',
            price_col=PRICE_COLUMNS[0],  # 価格_トヨタを使用
            change_lead_time=last_change_lt
        )
        
        # 結果メタデータを更新
        result_meta["revenue_actual"] = total_actual
        result_meta["revenue_predicted"] = total_predicted
        result_meta["revenue_difference"] = total_difference
        result_meta["success"] = True
        
        return predictions_result, result_meta
        
    except Exception as e:
        result_meta["error"] = str(e)
        return pd.DataFrame(), result_meta


def run_batch_prediction(
    data: pd.DataFrame,
    model,
    date_range: List[datetime.date],
    car_classes: List[str],
    model_metadata: Optional[Dict] = None,
    max_workers: int = 4
) -> Tuple[Dict[Tuple[datetime.date, str], pd.DataFrame], List[Dict[str, Any]]]:
    """複数の日付・車両クラス組み合わせに対してバッチ予測を実行

    Parameters
    ----------
    data : pd.DataFrame
        元データフレーム
    model
        予測モデル
    date_range : List[datetime.date]
        予測する利用日のリスト
    car_classes : List[str]
        対象車両クラスのリスト
    model_metadata : Optional[Dict], default=None
        モデルのメタデータ
    max_workers : int, default=4
        並列処理ワーカー数

    Returns
    -------
    Tuple[Dict[Tuple[datetime.date, str], pd.DataFrame], List[Dict[str, Any]]]
        (予測結果辞書, メタデータリスト)
    """
    predictions = {}
    metadata_list = []
    
    # 進捗バーの作成
    total_combinations = len(date_range) * len(car_classes)
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0
    
    # 並列処理用のタスクリスト作成
    tasks = []
    for date in date_range:
        for car_class in car_classes:
            tasks.append((date, car_class))
            
    # バッチ処理実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # タスク送信
        future_to_task = {
            executor.submit(batch_predict_date, data, model, date, car_class, model_metadata): (date, car_class)
            for date, car_class in tasks
        }
        
        # 結果回収
        for future in concurrent.futures.as_completed(future_to_task):
            date, car_class = future_to_task[future]
            try:
                prediction_df, meta = future.result()
                predictions[(date, car_class)] = prediction_df
                metadata_list.append(meta)
                
                # 進捗更新
                completed += 1
                progress_bar.progress(completed / total_combinations)
                status_text.text(f"処理中... {completed}/{total_combinations} 完了")
            except Exception as e:
                st.error(f"{date} {car_class}の処理中にエラー: {e}")
                metadata_list.append({
                    "date": date,
                    "car_class": car_class,
                    "success": False,
                    "error": str(e)
                })
    
    # 進捗表示をクリア
    status_text.empty()
    progress_bar.empty()
    
    return predictions, metadata_list


def display_batch_results(metadata_list: List[Dict[str, Any]]):
    """バッチ処理結果の集計表示

    Parameters
    ----------
    metadata_list : List[Dict[str, Any]]
        バッチ予測のメタデータリスト
    """
    if not metadata_list:
        st.warning("バッチ処理結果がありません")
        return
        
    # 成功件数・失敗件数を集計
    success_count = sum(1 for meta in metadata_list if meta.get("success", False))
    fail_count = len(metadata_list) - success_count
    
    st.metric("処理総数", f"{len(metadata_list)}件", f"成功: {success_count}件, 失敗: {fail_count}件")
    
    if success_count == 0:
        st.error("すべての処理が失敗しました")
        st.subheader("失敗詳細")
        error_df = pd.DataFrame([
            {"日付": meta.get("date"), "車両クラス": meta.get("car_class"), "エラー内容": meta.get("error", "不明")}
            for meta in metadata_list if not meta.get("success", False)
        ])
        st.dataframe(error_df)
        return
        
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
    
    fig = plot_batch_revenue_comparison(date_revenue_df, "利用日")
    st.plotly_chart(fig, use_container_width=True)
    
    # グラフ表示（車両クラスごとの集計）
    st.subheader("車両クラス別売上差額")
    class_revenue_df = result_df.groupby("車両クラス").agg({
        "実績売上": "sum",
        "予測売上": "sum",
        "売上差額": "sum"
    }).reset_index()
    
    fig2 = plot_batch_revenue_comparison(class_revenue_df, "車両クラス", horizontal=True)
    st.plotly_chart(fig2, use_container_width=True)
    
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