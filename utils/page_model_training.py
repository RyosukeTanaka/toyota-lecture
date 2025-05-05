# utils/page_model_training.py

import streamlit as st
import pandas as pd
from typing import Dict, Any
import datetime

from .constants import (
    TARGET_VARIABLE, DATE_COLUMN, CAR_CLASS_COLUMN,
    BOOKING_DATE_COLUMN, LEAD_TIME_COLUMN, USAGE_COUNT_COLUMN
)
from .modeling import setup_and_compare_models, get_feature_importance_df
from .visualization import plot_feature_importance
from .ui_components import render_model_training_sidebar_widgets
from .model_storage import save_model, list_saved_models, delete_model

def render_model_training_page(data: pd.DataFrame, config: Dict[str, Any]):
    st.title("モデルトレーニング")

    # --- サイドバーウィジェットの描画と値の取得 ---
    (
        selected_car_class,
        selected_numeric,
        selected_categorical,
        selected_features,
        models_to_compare,
        model_name,
        run_training
    ) = render_model_training_sidebar_widgets(data, config)

    # --- メインエリア --- #
    st.subheader("処理済みデータプレビュー")
    st.dataframe(data.head())

    st.markdown("---")

    # 保存済みモデル一覧表示
    st.header("保存済みモデル一覧")
    saved_models = list_saved_models()
    
    if saved_models:
        # モデル情報を表示するためのデータフレームを作成
        display_cols = ["model_name", "model_type", "car_class", "creation_date", "target_variable"]
        # メトリクス列を追加（存在する場合）
        if saved_models[0].get("metrics"):
            metric_keys = saved_models[0]["metrics"].keys()
            for model in saved_models:
                if "metrics" in model:
                    for key in metric_keys:
                        model[key] = model["metrics"].get(key)
            display_cols.extend(metric_keys)
        
        # 表示用データを整理
        models_display_data = []
        for model in saved_models:
            model_display = {col: model.get(col) for col in display_cols if col in model or col in model.get("metrics", {})}
            models_display_data.append(model_display)
        
        models_df = pd.DataFrame(models_display_data)
        st.dataframe(models_df)
        
        # モデル削除機能
        with st.expander("モデル削除"):
            models_to_delete = st.multiselect(
                "削除するモデルを選択",
                options=[model["model_name"] for model in saved_models],
                key="models_to_delete"
            )
            
            delete_button = st.button("選択したモデルを削除", key="delete_models_button")
            
            if delete_button and models_to_delete:
                for model_name_to_delete in models_to_delete:
                    for model in saved_models:
                        if model["model_name"] == model_name_to_delete:
                            success = delete_model(model["path"])
                            if success:
                                st.success(f"モデル '{model_name_to_delete}' を削除しました。")
                            else:
                                st.error(f"モデル '{model_name_to_delete}' の削除に失敗しました。")
                # 削除後に再表示
                st.rerun()
    else:
        st.info("保存済みモデルはありません。以下でモデルをトレーニングしてください。")

    st.markdown("---")

    # --- モデルトレーニング --- #
    st.header("新規モデルトレーニング")
    
    if run_training:
        if not model_name:
            st.warning("モデル名を入力してください。")
        elif not selected_features:
            st.warning("特徴量を選択してください。")
        elif not models_to_compare:
            st.warning("比較するモデルを選択してください。")
        else:
            st.markdown("---")
            st.subheader("モデルトレーニング実行")
            with st.spinner('モデルトレーニング中...'):
                data_for_modeling = data[data[CAR_CLASS_COLUMN] == selected_car_class] if selected_car_class != "全クラス" else data

                with st.expander("モデル学習に使用するデータのプレビュー"):
                    columns_to_show = selected_features.copy()
                    if TARGET_VARIABLE not in columns_to_show:
                        columns_to_show.append(TARGET_VARIABLE)
                    
                    existing_columns_to_show = [col for col in columns_to_show if col in data_for_modeling.columns]
                    if pd.Series(existing_columns_to_show).duplicated().any():
                        existing_columns_to_show = pd.Series(existing_columns_to_show).drop_duplicates().tolist()
                    
                    if existing_columns_to_show:
                        st.dataframe(data_for_modeling[existing_columns_to_show].head())
                    else:
                        st.warning("表示する列が見つかりません。")

                # 無視リスト生成
                potential_features = [col for col in data.columns if col not in [
                    TARGET_VARIABLE, DATE_COLUMN, BOOKING_DATE_COLUMN, 
                    LEAD_TIME_COLUMN, 'リードタイム', USAGE_COUNT_COLUMN
                ]]
                all_numeric_cols = data_for_modeling[potential_features].select_dtypes(include=['number']).columns.tolist()
                all_category_cols = data_for_modeling[potential_features].select_dtypes(exclude=['number', 'datetime', 'timedelta']).columns.tolist()
                ignored_numeric = list(set(all_numeric_cols) - set(selected_numeric))
                ignored_categorical = list(set(all_category_cols) - set(selected_categorical))
                explicitly_ignored = ['曜日_name', 'en_name']
                final_ignore_features = list(set(ignored_numeric + ignored_categorical + explicitly_ignored))

                # モデル比較
                best_model, comparison_results, setup_result = setup_and_compare_models(
                    _data=data_for_modeling, 
                    target=TARGET_VARIABLE,
                    numeric_features=selected_numeric, 
                    categorical_features=selected_categorical,
                    ignore_features=final_ignore_features, 
                    include_models=models_to_compare,
                    sort_metric='RMSE'
                )

                if best_model is not None and setup_result is not None:
                    # 特徴量重要度
                    st.markdown("---")
                    st.subheader(f"最良モデル ({type(best_model).__name__}) の特徴量重要度")
                    importance_df = get_feature_importance_df(best_model, setup_result)
                    if importance_df is not None and not importance_df.empty:
                        fig_importance = plot_feature_importance(importance_df)
                        if fig_importance: 
                            st.plotly_chart(fig_importance, use_container_width=True)
                        with st.expander("特徴量重要度データ"): 
                            st.dataframe(importance_df)
                    else: 
                        st.info("このモデルでは特徴量重要度を表示できません。")
                    
                    # モデル評価結果表示
                    st.markdown("---")
                    st.subheader("モデル評価比較結果")
                    st.dataframe(comparison_results)
                    
                    # 最良モデルの主要なメトリクスを抽出
                    best_model_row = comparison_results.iloc[0]
                    metrics = {
                        "MAE": best_model_row.get("MAE", None),
                        "MSE": best_model_row.get("MSE", None),
                        "RMSE": best_model_row.get("RMSE", None),
                        "R2": best_model_row.get("R2", None),
                        "RMSLE": best_model_row.get("RMSLE", None)
                    }
                    
                    # メトリクスをクリーンアップ（None値を削除）
                    metrics = {k: v for k, v in metrics.items() if v is not None}
                    
                    # モデルの保存
                    st.markdown("---")
                    st.subheader("モデルの保存")
                    
                    model_type = type(best_model).__name__
                    
                    # 学習時の特徴量情報を取得
                    model_columns = None
                    try:
                        # 1. PyCaretのパイプラインから直接取得を試みる
                        if hasattr(setup_result, 'pipeline') and hasattr(setup_result.pipeline, 'feature_names_in_'):
                            model_columns = list(setup_result.pipeline.feature_names_in_)
                        # 2. 前処理後の特徴量名を取得
                        elif hasattr(setup_result, 'X_train_transformed') and hasattr(setup_result.X_train_transformed, 'columns'):
                            model_columns = list(setup_result.X_train_transformed.columns)
                        # 3. get_configメソッドを使用
                        elif hasattr(setup_result, 'get_config'):
                            X_train_transformed = setup_result.get_config('X_train_transformed')
                            if X_train_transformed is not None and hasattr(X_train_transformed, 'columns'):
                                model_columns = list(X_train_transformed.columns)
                        
                        if model_columns:
                            st.success(f"学習時の特徴量列名 {len(model_columns)} 個を抽出しました")
                        else:
                            st.warning("学習時の特徴量列名を抽出できませんでした。予測時の特徴量変換が制限される可能性があります。")
                    except Exception as e:
                        st.warning(f"特徴量情報の抽出中にエラーが発生しました: {e}")
                        st.warning("学習時の特徴量情報が保存されないため、予測時の特徴量変換が制限される可能性があります。")
                    
                    # 保存を実行
                    try:
                        model_path = save_model(
                            model=best_model,
                            model_name=model_name,
                            model_type=model_type,
                            target_variable=TARGET_VARIABLE,
                            selected_features=selected_features,
                            car_class=selected_car_class,
                            comparison_results=comparison_results,
                            metrics=metrics,
                            model_columns=model_columns,
                            categorical_features=selected_categorical
                        )
                        st.success(f"モデル '{model_name}' を保存しました！")
                        st.info(f"保存先: {model_path}")
                    except Exception as e:
                        st.error(f"モデルの保存中にエラーが発生しました: {e}")
                
                elif best_model is None:
                    st.error("モデル比較に失敗しました。データやパラメータを確認してください。")
                else:
                    st.error("PyCaretセットアップに失敗しました。データやパラメータを確認してください。") 