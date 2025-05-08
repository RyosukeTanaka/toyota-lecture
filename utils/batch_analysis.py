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


# ★★★ 新しい実績売上計算ヘルパー関数 ★★★
def _calculate_actual_revenue_iterative(df_sorted_asc: pd.DataFrame, target_col: str, price_col: str, lt_col: str) -> float:
    """反復法で実績売上を計算する"""
    total_revenue = 0
    prev_usage = 0
    # 降順でループ
    for index, row in df_sorted_asc.sort_values(by=lt_col, ascending=False).iterrows():
        current_usage = row.get(target_col, 0)
        current_price = row.get(price_col, 0)
        # clip(lower=0)相当の処理
        new_bookings = (current_usage - prev_usage)
        if new_bookings > 0:
            total_revenue += new_bookings * current_price
        prev_usage = current_usage
    return total_revenue

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
        "model_name": "不明",
        "price_before_change": np.nan, # 変更前価格を初期化
        "price_after_change": np.nan,  # 変更後価格を初期化
        "actual_cumulative_usage_end": np.nan, # ★★★ 実績利用台数累積の最終値用 ★★★
        "predicted_cumulative_usage_end": np.nan, # ★★★ 予測利用台数累積の最終値用 ★★★
        # ★★★ 追加予約数用のキーを追加 ★★★
        "additional_actual_bookings": np.nan,
        "additional_predicted_bookings": np.nan,
        # ★★★ 新しいキーを追加 ★★★
        "actual_usage_at_change_lt": np.nan,
        # ★★★ Add keys for sales breakdown ★★★
        "revenue_actual_before": np.nan,
        "revenue_actual_after": np.nan,
        "revenue_predicted_after": np.nan,
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
        
        # ★★★ 実績利用台数累積の最終値を取得 ★★★
        actual_usage_end = np.nan
        if not data_filtered_sorted.empty and TARGET_VARIABLE in data_filtered_sorted.columns and LEAD_TIME_COLUMN in data_filtered_sorted.columns:
             # Ensure TARGET_VARIABLE is numeric before trying to get the value
             if pd.api.types.is_numeric_dtype(data_filtered_sorted[TARGET_VARIABLE]):
                 # Get row(s) with the minimum lead time
                 min_lt_data = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] == data_filtered_sorted[LEAD_TIME_COLUMN].min()]
                 if not min_lt_data.empty:
                     # Use the first row if multiple have the min lead time
                     final_usage_value = min_lt_data[TARGET_VARIABLE].iloc[0]
                     if pd.notna(final_usage_value):
                         actual_usage_end = final_usage_value
        result_meta["actual_cumulative_usage_end"] = actual_usage_end
        # ★★★ ここまで ★★★

        # --- ★★★ 主要価格列での変動チェックを先に行う ★★★ ---
        main_price_col = PRICE_COLUMNS[0] if PRICE_COLUMNS else None
        has_main_price_changed = False
        if main_price_col and main_price_col in data_filtered_sorted.columns:
            # 欠損値を除いた上でユニーク数をカウントし、1より大きい（つまり最低2つの異なる価格がある）場合を変動ありとみなす
            if data_filtered_sorted[main_price_col].dropna().nunique() > 1:
                has_main_price_changed = True
        
        if not has_main_price_changed:
            st.info(f"{date} {car_class}: 主要価格列 '{main_price_col}' に実質的な変動がなかったため、売上差額は0として扱います。")
            total_actual_revenue = 0
            # ★★★ Use iterative helper function for actual revenue ★★★
            if not data_filtered_sorted.empty and TARGET_VARIABLE in data_filtered_sorted.columns and \
               main_price_col and main_price_col in data_filtered_sorted.columns:
                data_for_calc = data_filtered_sorted[[TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN]].copy()
                data_for_calc[TARGET_VARIABLE] = pd.to_numeric(data_for_calc[TARGET_VARIABLE], errors='coerce').fillna(0)
                data_for_calc[main_price_col] = pd.to_numeric(data_for_calc[main_price_col], errors='coerce').fillna(0)
                if pd.api.types.is_numeric_dtype(data_for_calc[TARGET_VARIABLE]) and pd.api.types.is_numeric_dtype(data_for_calc[main_price_col]):
                    total_actual_revenue = _calculate_actual_revenue_iterative(
                        data_for_calc, TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN
                    )
                else: st.warning(f"{date} {car_class}: 実績売上計算列数値型エラー（変動なしケース）。実績売上0。") # 修正：メッセージ
            else: st.warning(f"{date} {car_class}: 実績売上計算データ不足（変動なしケース）。実績売上0。") # 修正：メッセージ

            result_meta["revenue_actual"] = total_actual_revenue
            result_meta["revenue_predicted"] = total_actual_revenue # Same as actual
            result_meta["revenue_difference"] = 0
            # ★★★ Set breakdown to NaN or full actual ★★★
            result_meta["revenue_actual_before"] = total_actual_revenue # No change, so all is 'before'
            result_meta["revenue_actual_after"] = 0
            result_meta["revenue_predicted_after"] = 0
            result_meta["success"] = True
            # ★★★ 価格変動なし時は予測累積も実績と同じ ★★★
            result_meta["predicted_cumulative_usage_end"] = actual_usage_end
            min_lt = 0
            if not data_filtered_sorted.empty and LEAD_TIME_COLUMN in data_filtered_sorted.columns:
                 min_lt_val = data_filtered_sorted[LEAD_TIME_COLUMN].min()
                 min_lt = min_lt_val if pd.notna(min_lt_val) else 0
            result_meta["last_change_lt"] = min_lt
            # 追加予約数は NaN のまま
            return pd.DataFrame(), result_meta
        # --- ★★★ 主要価格列変動チェックここまで ★★★ ---

        # 主要価格列に変動があった場合のみ、以下の処理に進む
        last_change_lt_for_scenario = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)
        
        if last_change_lt_for_scenario is None or last_change_lt_for_scenario == 0:
            # 主要価格列には変動があったが、全価格列での最終変更点が見つからなかったケース
            st.warning(f"{date} {car_class}: 主要価格には変動がありましたが、最終変更点が特定できないか当日(LT=0)だったため、売上差額は0として扱います。 (LT={last_change_lt_for_scenario})")
            total_actual_revenue_fallback = 0
            # ★★★ Use iterative helper function for actual revenue ★★★
            if not data_filtered_sorted.empty and TARGET_VARIABLE in data_filtered_sorted.columns and \
               main_price_col and main_price_col in data_filtered_sorted.columns:
                data_for_calc_fb = data_filtered_sorted[[TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN]].copy()
                data_for_calc_fb[TARGET_VARIABLE] = pd.to_numeric(data_for_calc_fb[TARGET_VARIABLE], errors='coerce').fillna(0)
                data_for_calc_fb[main_price_col] = pd.to_numeric(data_for_calc_fb[main_price_col], errors='coerce').fillna(0)
                if pd.api.types.is_numeric_dtype(data_for_calc_fb[TARGET_VARIABLE]) and pd.api.types.is_numeric_dtype(data_for_calc_fb[main_price_col]):
                   total_actual_revenue_fallback = _calculate_actual_revenue_iterative(
                       data_for_calc_fb, TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN
                   )

            result_meta["revenue_actual"] = total_actual_revenue_fallback
            result_meta["revenue_predicted"] = total_actual_revenue_fallback # Same as actual
            result_meta["revenue_difference"] = 0
            # ★★★ Set breakdown to NaN or full actual ★★★
            result_meta["revenue_actual_before"] = total_actual_revenue_fallback # Consider all as 'before'
            result_meta["revenue_actual_after"] = 0
            result_meta["revenue_predicted_after"] = 0
            result_meta["success"] = True
            # ★★★ このケースも予測累積は実績と同じ ★★★
            result_meta["predicted_cumulative_usage_end"] = actual_usage_end
            result_meta["last_change_lt"] = last_change_lt_for_scenario
            # 価格変更が当日または特定不可なので変更前後の価格はNaNのまま
            return pd.DataFrame(), result_meta
        
        else: # 価格変動があった場合 (従来の処理)
            result_meta["last_change_lt"] = last_change_lt_for_scenario
            
            # --- ★★★ 変更前後の価格を取得 ★★★ ---
            price_after = np.nan
            price_before = np.nan
            if main_price_col:
                # 変更後価格 (last_change_lt_for_scenario 時点)
                price_row_after = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] == last_change_lt_for_scenario]
                if not price_row_after.empty:
                    price_after = price_row_after[main_price_col].iloc[0]

                # 変更前価格 (last_change_lt_for_scenario より大きいリードタイムで最も近いもの)
                data_before_change = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] > last_change_lt_for_scenario]
                if not data_before_change.empty:
                    closest_lt_before = data_before_change[LEAD_TIME_COLUMN].min() # 最もLTが小さい=時間的に直前
                    price_row_before = data_before_change[data_before_change[LEAD_TIME_COLUMN] == closest_lt_before]
                    if not price_row_before.empty:
                        price_before = price_row_before[main_price_col].iloc[0]
            
            result_meta["price_before_change"] = price_before
            result_meta["price_after_change"] = price_after
            # --- ★★★ 価格取得ここまで ★★★ ---

            # シナリオデータを作成（価格変更点固定）
            data_scenario = create_scenario_data(
                data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN,
                scenario_type='last_change_fixed', change_lead_time=last_change_lt_for_scenario
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
                                transformed_data[col].fillna(col_mean if pd.notna(col_mean) else 0, inplace=True) # NaN mean -> fill 0
                        remaining_nan_cols = transformed_data.columns[transformed_data.isna().any()].tolist()
                        if remaining_nan_cols:
                            for col in remaining_nan_cols:
                                transformed_data[col].fillna(0, inplace=True)
                    
                    y_pred = model.predict(transformed_data)
                    if y_pred is not None: # ★★★ 追加: y_predがNoneでないことを確認 ★★★
                        y_pred = np.round(y_pred) # ★★★ 追加: 予測値を四捨五入 ★★★
                else:
                    # st.warning(f"{date} {car_class}のモデルメタデータがないため、直接予測を試みます。")
                    numeric_cols = X.select_dtypes(include=['number', 'bool']).columns
                    X_numeric = X[numeric_cols].copy() # SettingWithCopyWarning対策でcopy()
                    nan_cols_direct = X_numeric.columns[X_numeric.isna().any()].tolist()
                    if nan_cols_direct:
                        X_numeric.fillna(0, inplace=True)
                    y_pred = model.predict(X_numeric)
                    if y_pred is not None: # ★★★ 追加: y_predがNoneでないことを確認 ★★★
                        y_pred = np.round(y_pred) # ★★★ 追加: 予測値を四捨五入 ★★★
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
                                if y_pred is not None: y_pred = np.round(y_pred) # ★★★ 四捨五入 ★★★
                            else:
                                numeric_cols_fallback_else = X.select_dtypes(include=['number', 'bool']).columns
                                X_numeric_fallback_else = X[numeric_cols_fallback_else].copy()
                                X_numeric_fallback_else.fillna(0, inplace=True) # 欠損値処理
                                y_pred = model.predict(X_numeric_fallback_else)
                                if y_pred is not None: y_pred = np.round(y_pred) # ★★★ 四捨五入 ★★★
                        else:
                            numeric_cols_fallback_no_feat = X.select_dtypes(include=['number', 'bool']).columns
                            X_numeric_fallback_no_feat = X[numeric_cols_fallback_no_feat].copy()
                            X_numeric_fallback_no_feat.fillna(0, inplace=True) # 欠損値処理
                            y_pred = model.predict(X_numeric_fallback_no_feat)
                            if y_pred is not None: y_pred = np.round(y_pred) # ★★★ 四捨五入 ★★★
                except Exception as e2:
                    result_meta["error"] = f"予測失敗(e1:{str(e1)}, e2:{str(e2)})"
                    return pd.DataFrame(), result_meta
            
            if y_pred is None: # ★★★ y_predがNoneのままならエラー ★★★
                result_meta["error"] = "予測結果の生成に失敗しました (y_pred is None)"
                return pd.DataFrame(), result_meta
                
            predictions_result = data_scenario.copy()
            predictions_result['prediction_label'] = y_pred
            
            # ★★★ 予測利用台数累積の最終値を取得 ★★★
            predicted_usage_end = np.nan
            if not predictions_result.empty and 'prediction_label' in predictions_result.columns and LEAD_TIME_COLUMN in predictions_result.columns:
                if pd.api.types.is_numeric_dtype(predictions_result['prediction_label']):
                     min_lt_pred_data = predictions_result[predictions_result[LEAD_TIME_COLUMN] == predictions_result[LEAD_TIME_COLUMN].min()]
                     if not min_lt_pred_data.empty:
                         final_pred_value = min_lt_pred_data['prediction_label'].iloc[0]
                         if pd.notna(final_pred_value):
                              predicted_usage_end = final_pred_value
            result_meta["predicted_cumulative_usage_end"] = predicted_usage_end
            # ★★★ ここまで ★★★

            # --- 追加予約数の計算 --- 
            actual_usage_at_change = np.nan
            # ... (Find actual_usage_at_change - unrounded) ...
            usage_rows_after_or_at = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] >= last_change_lt_for_scenario]
            if not usage_rows_after_or_at.empty and TARGET_VARIABLE in usage_rows_after_or_at.columns:
                 if pd.api.types.is_numeric_dtype(usage_rows_after_or_at[TARGET_VARIABLE]):
                     closest_row_actual = usage_rows_after_or_at.loc[usage_rows_after_or_at[LEAD_TIME_COLUMN].idxmin()]
                     val_actual = closest_row_actual[TARGET_VARIABLE]
                     if pd.notna(val_actual):
                          actual_usage_at_change = val_actual # Keep unrounded for calculation
                          # ★★★ Store the rounded value in meta ★★★
                          result_meta["actual_usage_at_change_lt"] = round(actual_usage_at_change, 1)

            predicted_usage_at_change = np.nan
            # ... (Find predicted_usage_at_change - unrounded) ...
            if not predictions_result.empty and 'prediction_label' in predictions_result.columns and LEAD_TIME_COLUMN in predictions_result.columns:
                usage_rows_after_or_at_pred = predictions_result[predictions_result[LEAD_TIME_COLUMN] >= last_change_lt_for_scenario]
                if not usage_rows_after_or_at_pred.empty and 'prediction_label' in usage_rows_after_or_at_pred.columns:
                    if pd.api.types.is_numeric_dtype(usage_rows_after_or_at_pred['prediction_label']):
                         closest_row_pred = usage_rows_after_or_at_pred.loc[usage_rows_after_or_at_pred[LEAD_TIME_COLUMN].idxmin()]
                         val_pred = closest_row_pred['prediction_label']
                         if pd.notna(val_pred):
                             predicted_usage_at_change = val_pred # Keep unrounded for calculation

            # Calculate additional bookings (using unrounded values, then round result)
            if pd.notna(result_meta["actual_cumulative_usage_end"]) and pd.notna(actual_usage_at_change):
                add_actual = result_meta["actual_cumulative_usage_end"] - actual_usage_at_change
                result_meta["additional_actual_bookings"] = round(max(0, add_actual), 1)
            if pd.notna(result_meta["predicted_cumulative_usage_end"]) and pd.notna(predicted_usage_at_change):
                add_pred = result_meta["predicted_cumulative_usage_end"] - predicted_usage_at_change
                result_meta["additional_predicted_bookings"] = round(max(0, add_pred), 1)

            # ★★★ ここで早期リターンするケースを追加 ★★★
            # 変更点が見つからない or 当日変更の場合 -> 差額0
            if last_change_lt_for_scenario is None or last_change_lt_for_scenario == 0:
                # ... 実績売上計算 & 差額0で早期リターン ...
                total_actual_revenue_fallback = _calculate_actual_revenue_iterative(
                    data_filtered_sorted[[TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN]].copy(),
                    TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN
                )
                result_meta["revenue_actual"] = total_actual_revenue_fallback
                result_meta["revenue_predicted"] = total_actual_revenue_fallback
                result_meta["revenue_difference"] = 0
                result_meta["success"] = True
                result_meta["predicted_cumulative_usage_end"] = actual_usage_end # 実績と同じ
                result_meta["last_change_lt"] = last_change_lt_for_scenario
                # Addtl bookings NaN
                return pd.DataFrame(), result_meta

            # --- Step 1: Calculate add_actual using unrounded values ---
            add_actual = np.nan
            actual_usage_end_unrounded = result_meta["actual_cumulative_usage_end"] # Use the stored rounded value (should be fine for diff) or re-fetch unrounded if needed
            if pd.notna(actual_usage_end_unrounded) and pd.notna(actual_usage_at_change):
                add_actual = actual_usage_end_unrounded - actual_usage_at_change
                add_actual = max(0, add_actual) # Ensure non-negative

            # Store rounded add_actual regardless of the check outcome
            result_meta["additional_actual_bookings"] = round(add_actual, 1) if pd.notna(add_actual) else np.nan

            # --- Step 2: Check add_actual and return early if needed ---
            tolerance = 1e-9
            # Check if add_actual is calculated and is below tolerance
            if pd.notna(add_actual) and add_actual <= tolerance:
                st.info(f"価格変更後 (LT < {last_change_lt_for_scenario}) の追加実績予約数がほぼゼロのため、売上差額は0として扱います。", icon="ℹ️")
                # Calculate total actual revenue once more (full period)
                total_actual_revenue_no_add = _calculate_actual_revenue_iterative(
                    data_filtered_sorted[[TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN]].copy(),
                    TARGET_VARIABLE, main_price_col, LEAD_TIME_COLUMN
                )
                result_meta["revenue_actual"] = total_actual_revenue_no_add
                result_meta["revenue_predicted"] = total_actual_revenue_no_add # Predicted = Actual
                result_meta["revenue_difference"] = 0
                # Calculate and store add_pred even if returning early
                predicted_usage_at_change = np.nan
                # ... Find predicted_usage_at_change_unrounded ...
                if not predictions_result.empty and 'prediction_label' in predictions_result.columns and LEAD_TIME_COLUMN in predictions_result.columns:
                     usage_rows_after_or_at_pred = predictions_result[predictions_result[LEAD_TIME_COLUMN] >= last_change_lt_for_scenario]
                     if not usage_rows_after_or_at_pred.empty and 'prediction_label' in usage_rows_after_or_at_pred.columns:
                         if pd.api.types.is_numeric_dtype(usage_rows_after_or_at_pred['prediction_label']):
                              closest_row_pred = usage_rows_after_or_at_pred.loc[usage_rows_after_or_at_pred[LEAD_TIME_COLUMN].idxmin()]
                              val_pred = closest_row_pred['prediction_label']
                              if pd.notna(val_pred):
                                  predicted_usage_at_change = val_pred # unrounded
                add_pred = np.nan
                predicted_usage_end_unrounded = result_meta["predicted_cumulative_usage_end"] # Use stored rounded
                if pd.notna(predicted_usage_end_unrounded) and pd.notna(predicted_usage_at_change):
                     add_pred = predicted_usage_end_unrounded - predicted_usage_at_change
                     add_pred = max(0, add_pred)
                result_meta["additional_predicted_bookings"] = round(add_pred, 1) if pd.notna(add_pred) else np.nan
                result_meta["success"] = True
                return pd.DataFrame(), result_meta # ★★★ Return EARLY ★★★

            # --- Step 3: If add_actual > tolerance, calculate add_pred and call calculate_revenue_difference ---
            # add_actual is already stored and rounded above
            # Calculate and store add_pred
            predicted_usage_at_change = np.nan # Recalculate or ensure it's available
            # ... Find predicted_usage_at_change_unrounded ...
            if not predictions_result.empty and 'prediction_label' in predictions_result.columns and LEAD_TIME_COLUMN in predictions_result.columns:
                 usage_rows_after_or_at_pred = predictions_result[predictions_result[LEAD_TIME_COLUMN] >= last_change_lt_for_scenario]
                 if not usage_rows_after_or_at_pred.empty and 'prediction_label' in usage_rows_after_or_at_pred.columns:
                     if pd.api.types.is_numeric_dtype(usage_rows_after_or_at_pred['prediction_label']):
                          closest_row_pred = usage_rows_after_or_at_pred.loc[usage_rows_after_or_at_pred[LEAD_TIME_COLUMN].idxmin()]
                          val_pred = closest_row_pred['prediction_label']
                          if pd.notna(val_pred):
                              predicted_usage_at_change = val_pred # unrounded
            add_pred = np.nan
            predicted_usage_end_unrounded = result_meta["predicted_cumulative_usage_end"] # Use stored rounded
            if pd.notna(predicted_usage_end_unrounded) and pd.notna(predicted_usage_at_change):
                 add_pred = predicted_usage_end_unrounded - predicted_usage_at_change
                 add_pred = max(0, add_pred)
            result_meta["additional_predicted_bookings"] = round(add_pred, 1) if pd.notna(add_pred) else np.nan

            # Call calculate_revenue_difference ONLY if add_actual was > tolerance
            (revenue_df,
             total_actual, total_predicted_hybrid, total_difference,
             actual_before, actual_after, predicted_after
            ) = calculate_revenue_difference(
                 df_actual=data_filtered_sorted,
                 df_predicted=predictions_result,
                 lead_time_col=LEAD_TIME_COLUMN,
                 actual_usage_col=TARGET_VARIABLE,
                 pred_usage_col='prediction_label',
                 price_col=main_price_col,
                 change_lead_time=last_change_lt_for_scenario
            )

            # ★★★ Ensure all breakdown values are stored in meta ★★★
            result_meta["revenue_actual"] = total_actual
            result_meta["revenue_predicted"] = total_predicted_hybrid
            result_meta["revenue_difference"] = total_difference
            result_meta["revenue_actual_before"] = actual_before # Store value from function
            result_meta["revenue_actual_after"] = actual_after   # Store value from function
            result_meta["revenue_predicted_after"] = predicted_after # Store value from function
            result_meta["success"] = True
            return predictions_result, result_meta
        
    except Exception as e:
        result_meta["error"] = str(e)
        import traceback
        st.error(f"batch_predict_dateでエラー: {e}\n{traceback.format_exc()}") # トレースバック表示を追加
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