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
# visualizationのインポートは display_batch_results が削除されるため不要になるか確認
# from .visualization import plot_batch_revenue_comparison # display_batch_results内で使われていた


def batch_predict_date(
    data: pd.DataFrame,
    models: Dict[str, Any],  # モデル辞書（車両クラス→モデル）
    date: datetime.date,
    car_class: str,
    models_metadata: Optional[Dict[str, Dict]] = None  # モデルメタデータ辞書（車両クラス→メタデータ）
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """単一の日付とクラスの組み合わせに対して予測を実行する

    Parameters
    ----------
    data : pd.DataFrame
        元データフレーム
    models : Dict[str, Any]
        予測モデル辞書（車両クラス→モデル）
    date : datetime.date
        予測する利用日
    car_class : str
        車両クラス
    models_metadata : Optional[Dict[str, Dict]], default=None
        モデルのメタデータ辞書（車両クラス→メタデータ）

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
        "revenue_difference": 0,
        "model_name": "不明"  # 使用したモデル名を記録
    }
    
    try:
        # 使用するモデルとメタデータを決定
        if car_class in models:
            # 車両クラス専用モデルがある場合はそれを使用
            model = models[car_class]
            model_metadata = models_metadata.get(car_class) if models_metadata else None
            # モデル名を記録
            if models_metadata and car_class in models_metadata and models_metadata[car_class]:
                result_meta["model_name"] = models_metadata[car_class].get("model_name", "不明")
        elif "全クラス" in models:
            # 全クラス用モデルをフォールバックとして使用
            model = models["全クラス"]
            model_metadata = models_metadata.get("全クラス") if models_metadata else None
            # モデル名を記録
            if models_metadata and "全クラス" in models_metadata and models_metadata["全クラス"]:
                result_meta["model_name"] = models_metadata["全クラス"].get("model_name", "不明") + " (全クラス)"
        else:
            result_meta["error"] = f"{car_class}用のモデルが見つかりません"
            return pd.DataFrame(), result_meta
        
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
            # 価格変更点が見つからない場合でも、エラーとせず処理を続行できるように変更
            # result_meta["error"] = "価格変更点が見つかりません"
            # return pd.DataFrame(), result_meta
            st.info(f"{date} {car_class}: 価格変更点が見つかりませんでした。全期間を対象とします。")
            # 価格変更点がない場合は、リードタイムの最小値（0など）を使うか、特別な処理を検討
            # ここでは、最も小さいリードタイムを使用する（例：0日）
            last_change_lt = data_filtered_sorted[LEAD_TIME_COLUMN].min()
            if pd.isna(last_change_lt):
                last_change_lt = 0 # それでもNaNなら0
            
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
        # `model_metadata` や `model_columns` がNoneでないことを確認
        model_cols_in_meta = model_metadata.get("model_columns", []) if model_metadata else []
        for col in object_cols:
            if model_metadata and "model_columns" in model_metadata and col in model_cols_in_meta:
                continue
            else:
                scen_data_transformed = scen_data_transformed.drop(columns=[col])
        
        if TARGET_VARIABLE in scen_data_transformed.columns:
            X = scen_data_transformed.drop(columns=[TARGET_VARIABLE])
        else:
            X = scen_data_transformed
            
        y_pred = None
        
        try:
            if model_metadata and "model_columns" in model_metadata:
                # st.info(f"{date} {car_class}の特徴量変換を適用...（モデル: {result_meta['model_name']}）")
                transformed_data = prepare_features_for_prediction(X, model_metadata)
                
                # 特徴量の不一致チェックと修正
                model_feature_names = model_metadata.get("model_columns", [])
                missing_features = [col for col in model_feature_names if col not in transformed_data.columns]
                if missing_features:
                    # st.warning(f"変換後も足りない特徴量があります: {missing_features}")
                    for col in missing_features:
                        transformed_data[col] = 0
                
                extra_features = [col for col in transformed_data.columns if col not in model_feature_names]
                if extra_features:
                    # st.warning(f"モデルに不要な特徴量を削除します: {extra_features}")
                    transformed_data = transformed_data.drop(columns=extra_features)
                
                if model_feature_names: # model_feature_namesが空でないことを確認
                    transformed_data = transformed_data[model_feature_names]
                
                cat_cols = transformed_data.select_dtypes(include=['object']).columns
                if not cat_cols.empty:
                    # st.warning(f"変換後もobject型の列が残っています: {list(cat_cols)}。数値型に変換します。")
                    for col in cat_cols:
                        transformed_data[col] = pd.factorize(transformed_data[col])[0]
                
                nan_cols = transformed_data.columns[transformed_data.isna().any()].tolist()
                if nan_cols:
                    # st.warning(f"予測データに欠損値(NaN)が含まれています。欠損値を処理します: {nan_cols}")
                    numeric_cols_nan = transformed_data.select_dtypes(include=['number']).columns
                    nan_numeric_cols = [col for col in nan_cols if col in numeric_cols_nan]
                    if nan_numeric_cols:
                        for col in nan_numeric_cols:
                            col_mean = transformed_data[col].mean()
                            transformed_data[col].fillna(col_mean, inplace=True)
                    remaining_nan_cols = transformed_data.columns[transformed_data.isna().any()].tolist()
                    if remaining_nan_cols:
                        for col in remaining_nan_cols:
                            transformed_data[col].fillna(0, inplace=True)
                
                y_pred = model.predict(transformed_data)
            else:
                # st.warning(f"{date} {car_class}のモデルメタデータがないため、直接予測を試みます。")
                numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
                X_numeric = X[numeric_cols].copy() # SettingWithCopyWarning対策でcopy()
                nan_cols_direct = X_numeric.columns[X_numeric.isna().any()].tolist()
                if nan_cols_direct:
                    X_numeric.fillna(0, inplace=True)
                y_pred = model.predict(X_numeric)
        except Exception as e1:
            try:
                if hasattr(model, 'predict'):
                    # st.info(f"{date} {car_class}のフォールバック予測を試行...")
                    model_features_direct = []
                    if hasattr(model, 'feature_names_in_'):
                        model_features_direct = list(model.feature_names_in_)
                    
                    if model_features_direct:
                        numeric_X_fallback = X.select_dtypes(include=['number', 'bool'])
                        available_features_fallback = [col for col in model_features_direct if col in numeric_X_fallback.columns]
                        if available_features_fallback:
                            X_selected_fallback = numeric_X_fallback[available_features_fallback].copy()
                            missing_features_fallback = [col for col in model_features_direct if col not in X_selected_fallback.columns]
                            for col in missing_features_fallback:
                                X_selected_fallback[col] = 0
                            X_selected_fallback = X_selected_fallback[model_features_direct]
                            nan_cols_fallback = X_selected_fallback.columns[X_selected_fallback.isna().any()].tolist()
                            if nan_cols_fallback:
                                for col in nan_cols_fallback:
                                    col_mean_fallback = X_selected_fallback[col].mean()
                                    if pd.isna(col_mean_fallback):
                                        X_selected_fallback[col].fillna(0, inplace=True)
                                    else:
                                        X_selected_fallback[col].fillna(col_mean_fallback, inplace=True)
                            if X_selected_fallback.isna().any().any():
                                X_selected_fallback.fillna(0, inplace=True)
                            y_pred = model.predict(X_selected_fallback)
                        else:
                            numeric_cols_fallback_else = X.select_dtypes(include=['number', 'bool']).columns
                            X_numeric_fallback_else = X[numeric_cols_fallback_else].copy()
                            X_numeric_fallback_else.fillna(0, inplace=True) # 欠損値処理
                            y_pred = model.predict(X_numeric_fallback_else)
                    else:
                        numeric_cols_fallback_no_feat = X.select_dtypes(include=['number', 'bool']).columns
                        X_numeric_fallback_no_feat = X[numeric_cols_fallback_no_feat].copy()
                        X_numeric_fallback_no_feat.fillna(0, inplace=True) # 欠損値処理
                        y_pred = model.predict(X_numeric_fallback_no_feat)
            except Exception as e2:
                result_meta["error"] = f"予測失敗(e1:{str(e1)}, e2:{str(e2)})"
                return pd.DataFrame(), result_meta
                
        if y_pred is None:
            result_meta["error"] = "予測結果がNoneです"
            return pd.DataFrame(), result_meta
            
        predictions_result = data_scenario.copy()
        predictions_result['prediction_label'] = y_pred
        
        revenue_df, total_actual, total_predicted, total_difference = calculate_revenue_difference(
            df_actual=data_filtered_sorted,
            df_predicted=predictions_result,
            lead_time_col=LEAD_TIME_COLUMN,
            actual_usage_col=TARGET_VARIABLE,
            pred_usage_col='prediction_label',
            price_col=PRICE_COLUMNS[0], 
            change_lead_time=last_change_lt
        )
        
        result_meta["revenue_actual"] = total_actual
        result_meta["revenue_predicted"] = total_predicted
        result_meta["revenue_difference"] = total_difference
        result_meta["success"] = True
        
        return predictions_result, result_meta
        
    except Exception as e:
        result_meta["error"] = str(e)
        # import traceback # デバッグ時のみ
        # traceback.print_exc() # デバッグ時のみ
        return pd.DataFrame(), result_meta


def run_batch_prediction(
    data: pd.DataFrame,
    models: Dict[str, Any],  # モデル辞書に変更
    date_range: List[datetime.date],
    car_classes: List[str],
    models_metadata: Optional[Dict[str, Dict]] = None,  # モデルメタデータ辞書に変更
    max_workers: int = 4
) -> Tuple[Dict[Tuple[datetime.date, str], pd.DataFrame], List[Dict[str, Any]]]:
    """複数の日付・車両クラス組み合わせに対してバッチ予測を実行

    Parameters
    ----------
    data : pd.DataFrame
        元データフレーム
    models : Dict[str, Any]
        車両クラスごとの予測モデル辞書
    date_range : List[datetime.date]
        予測する利用日のリスト
    car_classes : List[str]
        対象車両クラスのリスト
    models_metadata : Optional[Dict[str, Dict]], default=None
        車両クラスごとのモデルメタデータ辞書
    max_workers : int, default=4
        並列処理ワーカー数

    Returns
    -------
    Tuple[Dict[Tuple[datetime.date, str], pd.DataFrame], List[Dict[str, Any]]]
        (予測結果辞書, メタデータリスト)
    """
    predictions = {}
    metadata_list = []
    
    total_combinations = len(date_range) * len(car_classes)
    # Streamlit要素はメインスレッドからのみ呼び出し可能なので、ここではコメントアウト
    # progress_bar = st.progress(0)
    # status_text = st.empty()
    completed_tasks = 0 # completed だとPythonの予約語と被るので変更
    
    tasks = [(date_item, car_class_item) for date_item in date_range for car_class_item in car_classes]
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(batch_predict_date, data, models, task_date, task_car_class, models_metadata):
            (task_date, task_car_class) for task_date, task_car_class in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_date_done, task_car_class_done = future_to_task[future]
            try:
                prediction_df, meta = future.result()
                predictions[(task_date_done, task_car_class_done)] = prediction_df
                metadata_list.append(meta)
                
                completed_tasks += 1
                # progress_bar.progress(completed_tasks / total_combinations)
                # status_text.text(f"処理中... {completed_tasks}/{total_combinations} 完了")
                print(f"Batch processing: {completed_tasks}/{total_combinations} done - {task_date_done} {task_car_class_done} - Success: {meta.get('success')}") # コンソールに進捗表示
            except Exception as e_task:
                print(f"Error processing {task_date_done} {task_car_class_done}: {e_task}") # コンソールにエラー表示
                metadata_list.append({
                    "date": task_date_done,
                    "car_class": task_car_class_done,
                    "success": False,
                    "error": str(e_task),
                    "model_name": models_metadata.get(task_car_class_done, {}).get("model_name", "不明") if models_metadata else "不明"
                })
                completed_tasks += 1 # エラーでもタスクは完了したとみなす
                # progress_bar.progress(completed_tasks / total_combinations)
                # status_text.text(f"処理中... {completed_tasks}/{total_combinations} 完了 (エラー発生)")
    
    # status_text.empty()
    # progress_bar.empty()
    
    return predictions, metadata_list


# display_batch_results 関数は utils/page_batch_analysis.py に display_batch_results_in_page として移植されたため、ここでは削除します。
# 以前の display_batch_results 関数のコードは削除済みです。 