# 機械学習モデル関連の関数
from pycaret.regression import setup, compare_models, predict_model, pull, plot_model
import streamlit as st
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import numpy as np # numpyをインポート
from typing import Tuple, List, Dict, Any, Optional, Union # 型ヒント用に追加

# グローバル変数やセッションステートでモデルを保持することも検討できるが、
# ここでは関数間でモデルオブジェクトを渡すシンプルな方法を採用。

# @st.cache_resource # 一時的にキャッシュを無効化
def setup_and_compare_models(_data, target, numeric_features, categorical_features, ignore_features, include_models, sort_metric='RMSE', fold=5):
    """PyCaretのセットアップとモデル比較を実行し、最良モデル、結果、セットアップオブジェクトを返す"""
    st.info("PyCaretのセットアップを開始します...")
    setup_result = None # 初期化
    try:
        # ★★★ 追加: ターゲット変数が欠損している行を削除 ★★★
        original_rows = len(_data)
        _data_cleaned = _data.dropna(subset=[target]).copy()
        removed_rows = original_rows - len(_data_cleaned)
        if removed_rows > 0:
             st.warning(f"ターゲット変数 '{target}' の欠損値を含む {removed_rows} 行をモデル学習データから除外しました。")
        # ★★★ ---------------------------------------- ★★★

        # html=FalseにしないとStreamlit内で表示がおかしくなることがある
        # ★★★ setup に欠損値補完の引数を追加 ★★★
        setup_result = setup(data=_data_cleaned, target=target,
                       numeric_features=numeric_features if numeric_features else None,
                       categorical_features=categorical_features if categorical_features else None,
                       ignore_features=ignore_features if ignore_features else None,
                       numeric_imputation='mean', # 数値特徴量のNaNは平均値で補完
                       categorical_imputation='mode', # カテゴリ特徴量のNaNは最頻値で補完
                       session_id=123, # 再現性のため
                       verbose=False, # Streamlit上では詳細ログを抑制
                       html=False,
                       fold_strategy='timeseries', # 時系列クロスバリデーションを指定
                       fold=fold,
                       data_split_shuffle=False, # ★ 追加: 時系列CVでは必須
                       fold_shuffle=False)     # ★ 追加: 時系列CVでは必須
        st.success("PyCaret セットアップ完了！")

        st.info(f"選択されたモデル: {', '.join(include_models)} で比較を実行します (fold_strategy='timeseries', fold={fold}, sort='{sort_metric}')...")
        best_model = compare_models(include=include_models, fold=fold, sort=sort_metric,
                                    verbose=False)
        comparison_results = pull()
        st.success("モデル比較完了！")
        st.write("最良モデル:", best_model)

        # --- ★ 追加: 詳細なCV結果を表示 ★ ---
        try:
            st.markdown("---")
            st.subheader("クロスバリデーション詳細結果 (各Foldのスコア)")
            # PyCaretのバージョンや内部構造により属性名が異なる可能性あり
            if hasattr(setup_result, 'cv_results_'):
                 detailed_cv_results = setup_result.cv_results_
                 if isinstance(detailed_cv_results, pd.DataFrame):
                     st.dataframe(detailed_cv_results)
                 elif isinstance(detailed_cv_results, dict): # 辞書の場合もある
                     st.dataframe(pd.DataFrame(detailed_cv_results))
                 else:
                     st.write("CV詳細結果が予期せぬ形式です:", type(detailed_cv_results))
            elif hasattr(setup_result, 'results_'): # 別の可能性
                 detailed_cv_results = setup_result.results_
                 # ... (同様に表示処理) ...
                 if isinstance(detailed_cv_results, pd.DataFrame):
                     st.dataframe(detailed_cv_results)
                 elif isinstance(detailed_cv_results, dict):
                     st.dataframe(pd.DataFrame(detailed_cv_results))
                 else:
                     st.write("CV詳細結果が予期せぬ形式です:", type(detailed_cv_results))
            else:
                 st.warning("セットアップオブジェクトからCV詳細結果属性 (`cv_results_`など) が見つかりませんでした。")
                 st.write("利用可能な属性:", dir(setup_result)) # デバッグ用

        except Exception as e_cv_detail:
             st.error(f"CV詳細結果の表示中にエラーが発生しました: {e_cv_detail}")
        # ------------------------------------

        # UIには pull() の集計結果を引き続き返す
        return best_model, comparison_results, setup_result

    except Exception as e:
        st.error(f"PyCaret処理中にエラーが発生しました: {e}")
        st.error("ターゲット変数、特徴量の選択、データ型などを確認してください。")
        # traceback.print_exc()
        # セットアップ結果も含めてNoneや空のDataFrameを返す
        return None, pd.DataFrame(), setup_result


# ★★★ 戻り値の型ヒントを修正 ★★★
def predict_with_model(model, _data, target: str) -> Tuple[pd.DataFrame, Optional[List[Dict]], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """与えられたモデル、データ、ターゲット列名で予測を実行し、予測結果DF、補完ログ、NaNがあった行(補完前)、NaNがあった行(補完後)を返す"""
    imputation_log: List[Dict] = [] # 補完ログを初期化
    nan_rows_before_imputation: Optional[pd.DataFrame] = None # NaNがあった行(補完前)
    nan_rows_after_imputation: Optional[pd.DataFrame] = None  # NaNがあった行(補完後)
    nan_rows_mask = None # マスクを保持する変数

    if model is None or _data.empty:
        st.error("予測に必要なモデルまたはデータがありません。")
        return pd.DataFrame(), None, None, None # 空DFとNoneを返す
    try:
        # --- 予測データからターゲット列を削除 ---
        data_for_prediction = _data.copy()
        if target in data_for_prediction.columns:
            data_for_prediction = data_for_prediction.drop(columns=[target])

        # ★★★ 予測データの特徴量に含まれるNaNを強制的に補完 & ログ記録 ★★★
        st.info("予測データの特徴量のNaNを補完しています...")
        original_nan_counts = data_for_prediction.isnull().sum()
        cols_with_nan = original_nan_counts[original_nan_counts > 0].index.tolist()

        if cols_with_nan:
            st.warning(f"予測データの特徴量でNaNが見つかりました。補完を実行します: {cols_with_nan}")

            # ★★★ NaNを含む行のマスクを作成し、補完前のデータを保存 ★★★
            nan_rows_mask = data_for_prediction[cols_with_nan].isnull().any(axis=1)
            if nan_rows_mask.any():
                 nan_rows_before_imputation = data_for_prediction[nan_rows_mask].copy()
            # ★★★ ------------------------------------------ ★★★

            for col in cols_with_nan:
                # NaNカウントを補完前に取得
                nan_count_for_col = data_for_prediction[col].isnull().sum()
                imputation_method = "unknown"
                imputation_value: Any = "error"
                if pd.api.types.is_numeric_dtype(data_for_prediction[col]):
                    mean_val = data_for_prediction[col].mean()
                    data_for_prediction[col] = data_for_prediction[col].fillna(mean_val)
                    imputation_method = "mean"
                    imputation_value = mean_val
                elif pd.api.types.is_object_dtype(data_for_prediction[col]) or pd.api.types.is_categorical_dtype(data_for_prediction[col]):
                    mode_val = data_for_prediction[col].mode()
                    if not mode_val.empty:
                        fill_value = mode_val[0]
                        data_for_prediction[col] = data_for_prediction[col].fillna(fill_value)
                        imputation_method = "mode"
                        imputation_value = fill_value
                    else:
                        data_for_prediction[col] = data_for_prediction[col].fillna('missing')
                        imputation_method = "missing_value_string"
                        imputation_value = "missing"
                else:
                     data_for_prediction[col] = data_for_prediction[col].fillna('missing_unknown_type')
                     imputation_method = "missing_unknown_type"
                     imputation_value = "missing_unknown_type"

                # ログに追加 (Affected Rows を追加)
                imputation_log.append({
                    "Column": col,
                    "Affected Rows": nan_count_for_col,
                    "Imputation Method": imputation_method,
                    "Imputation Value": imputation_value
                })

            # ★★★ 補完後のNaNがあった行を保存 ★★★
            if nan_rows_mask is not None and nan_rows_mask.any():
                 nan_rows_after_imputation = data_for_prediction[nan_rows_mask].copy()
            # ★★★ -------------------------- ★★★

            final_nan_counts = data_for_prediction.isnull().sum().sum()
            if final_nan_counts == 0:
                 st.success("予測データの特徴量のNaN補完が完了しました。")
            else:
                 st.error(f"NaNの補完後も {final_nan_counts} 個のNaNが残っています。処理を確認してください。")
                 st.dataframe(data_for_prediction[data_for_prediction.isnull().any(axis=1)])
                 # ★★★ エラー時もログとNaN行データ(補完前後)は返す ★★★
                 return pd.DataFrame(), imputation_log, nan_rows_before_imputation, nan_rows_after_imputation
        else:
             st.info("予測データの特徴量にNaNはありませんでした。")
        # ★★★ ----------------------------------------------- ★★★

        st.info("予測を実行しています...")
        
        # PyCaretのpredict_modelを使用せず、モデルを直接使用（保存・読み込み後の互換性のため）
        try:
            # モデルがパイプラインの場合
            if hasattr(model, 'predict'):
                # 直接予測を実行
                predictions = model.predict(data_for_prediction)
                
                # 結果をデータフレームに変換
                predictions_result = _data.copy()
                predictions_result['prediction_label'] = predictions
                
                st.success("予測完了！")
                return predictions_result, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation
            else:
                st.error("モデルにpredict()メソッドがありません。モデルの形式を確認してください。")
                return pd.DataFrame(), imputation_log, nan_rows_before_imputation, nan_rows_after_imputation
        except Exception as predict_error:
            st.error(f"predict()メソッドでの予測中にエラーが発生しました: {predict_error}")
            # PyCaret方式をフォールバックとして試行
            try:
                st.warning("PyCaret predict_model()を試行します...")
                predictions_result = predict_model(model, data=data_for_prediction)
                st.success("PyCaret方式での予測完了！")
                
                # --- 結果を元のデータフレーム (_data) と結合して返す ---
                predictions_final = _data.copy()
                if predictions_final.index.equals(predictions_result.index):
                     predictions_final['prediction_label'] = predictions_result['prediction_label']
                else:
                     st.warning("予測結果と元のデータのインデックスが一致しません。インデックスに基づいてマージします。")
                     predictions_final = pd.merge(
                         predictions_final,
                         predictions_result[['prediction_label']],
                         left_index=True,
                         right_index=True,
                         how='left'
                     )
                     if predictions_final['prediction_label'].isnull().any():
                          st.warning("予測結果のマージ後、一部の行で予測値がNaNになりました。インデックス不一致の可能性があります。")

                return predictions_final, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation
            except Exception as pycaret_error:
                st.error(f"PyCaretでの予測中にもエラーが発生しました: {pycaret_error}")
                return pd.DataFrame(), imputation_log, nan_rows_before_imputation, nan_rows_after_imputation

    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        # ★★★ エラー時もログとNaN行データ(補完前後)を返す ★★★
        return pd.DataFrame(), imputation_log, nan_rows_before_imputation, nan_rows_after_imputation

# 特徴量重要度プロットを取得する関数を追加
# @st.cache_data # モデルオブジェクト自体はキャッシュされているので、ここは都度実行でも良いかも
def get_feature_importance_plot(model):
    """与えられたモデルの特徴量重要度プロット(matplotlib)を返す"""
    if model is None:
        return None
    try:
        st.info("特徴量重要度を生成中...")
        # plot_modelは内部でmatplotlibを使用し、設定によってはファイルに保存したり表示したりする
        # ここではプロットオブジェクト自体を取得したいが、直接返さない場合がある
        # 回避策として、プロットを表示させ、それをキャプチャする
        # 重要: PyCaretのバージョンや環境によっては動作が異なる可能性あり

        # 一時的にmatplotlibのインタラクティブモードをオフにするなどが必要な場合も
        # plt.ioff()

        # plot_modelを呼び出し、現在のFigureを取得する
        # `display_format='streamlit'` は非推奨になった可能性あり。デフォルトで良いはず。
        plot_model(model, plot='feature', display_format=None) # display_format=None で表示を試みる
        fig = plt.gcf() # 現在のFigureを取得

        # プロット後にクリアするかどうか -> Streamlit側で管理されるはずなので不要
        # plt.clf()
        # plt.close(fig)

        if fig and fig.axes:
            st.success("特徴量重要度の生成完了！")
            return fig # Figureオブジェクトを返す
        else:
            st.warning("特徴量重要度プロットの取得に失敗しました。モデルがサポートしていない可能性があります。")
            return None

    except Exception as e:
        st.error(f"特徴量重要度の生成中にエラーが発生しました: {e}")
        st.error(f"使用されたモデル ({type(model).__name__}) が特徴量重要度をサポートしていない可能性があります。")
        # traceback.print_exc()
        # エラーが発生した場合もNoneを返す
        if 'not supported' in str(e).lower(): # エラーメッセージに基づいて判定強化
            st.warning(f"モデル '{type(model).__name__}' は特徴量重要度プロットをサポートしていません。")
        return None
    # finallyブロックを削除 (または plt.close('all') をコメントアウト)
    # finally:
    #     plt.close('all')

# 特徴量重要度をDataFrameで取得する関数 (引数を変更)
def get_feature_importance_df(model, setup_result):
    """訓練済みモデル(最終推定器)とPyCaretのセットアップ結果から特徴量重要度を抽出し、DataFrameで返す"""
    # modelは最終推定器(e.g., XGBRegressor)である可能性が高いと想定
    # setup_resultは setup() の戻り値 (RegressionExperiment)
    if model is None or setup_result is None:
        st.warning("重要度抽出: モデルまたはセットアップ結果がありません。")
        return None

    try:
        st.info("特徴量重要度データを抽出中...")

        # --- 最終的なモデル推定器を取得 --- #
        if hasattr(model, 'steps'):
            final_estimator = model.steps[-1][1]
        else:
            st.warning("モデルがPipelineではないようです。そのままモデルを使用します。") # この警告は出る可能性がある
            final_estimator = model

        # --- 重要度を取得 --- #
        if hasattr(final_estimator, 'feature_importances_'):
            importances = final_estimator.feature_importances_
        else:
            # Pipelineだった場合も考慮して最終ステップを確認 (念のため)
            if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
                 final_estimator = model.steps[-1][1]
                 importances = final_estimator.feature_importances_
                 st.info("モデルはPipelineでしたが、最終推定器から重要度を取得しました。")
            else:
                 st.error(f"モデル ({type(model).__name__}) から feature_importances_ 属性が見つかりません。")
                 return None

        # --- 前処理後の特徴量名を取得 (修正箇所) --- #
        feature_names = None
        err_msg = [] # エラー収集用

        try:
            # 試行1: setup_result.pipeline から get_feature_names_out (推奨)
            # 注意: modelではなく、setup_result.pipeline を使う！
            if hasattr(setup_result, 'pipeline') and hasattr(setup_result.pipeline[:-1], 'get_feature_names_out'):
                 try:
                     # モデルステップを除いたパイプラインから名前を取得
                     feature_names = list(setup_result.pipeline[:-1].get_feature_names_out())
                     st.info("setup_result.pipeline[:-1].get_feature_names_out() から特徴量名を取得。")
                 except Exception as e1:
                     err_msg.append(f"pipeline.get_feature_names_out エラー: {e1}")

            # 試行2: get_config を使用
            if feature_names is None and hasattr(setup_result, 'get_config'):
                try:
                    # 直接的な属性がないか確認
                    names_transformed = setup_result.get_config('feature_names_transformed')
                    if names_transformed is not None:
                         feature_names = names_transformed
                         st.info("get_config('feature_names_transformed') から特徴量名を取得。")
                    else:
                         # X_transformedのカラム名を使用
                         X_transformed_df = setup_result.get_config('X_train_transformed') # または 'X_transformed'
                         if X_transformed_df is not None and hasattr(X_transformed_df, 'columns'):
                             feature_names = X_transformed_df.columns.tolist()
                             st.info("get_config('X_train_transformed').columns から特徴量名を取得。")
                         else:
                              err_msg.append("get_config から特徴量名が見つかりません。")
                except Exception as e2:
                     err_msg.append(f"get_configエラー: {e2}")

            # 試行3: 古い属性を直接参照
            if feature_names is None and hasattr(setup_result, 'feature_names_transformed'):
                 try:
                     feature_names = setup_result.feature_names_transformed
                     st.info("setup_result.feature_names_transformed 属性から特徴量名を取得。")
                 except Exception as e3:
                      err_msg.append(f"feature_names_transformed属性エラー: {e3}")

            if feature_names is None and hasattr(setup_result, 'X_train_transformed'):
                 try:
                     feature_names = setup_result.X_train_transformed.columns.tolist()
                     st.info("setup_result.X_train_transformed.columns 属性から特徴量名を取得。")
                 except Exception as e4:
                     err_msg.append(f"X_train_transformed.columns属性エラー: {e4}")


            # --- 長さチェックと調整 --- #
            if feature_names is not None:
                if len(feature_names) == len(importances):
                    st.success("特徴量名と重要度の数が一致しました。")
                else:
                    st.warning(f"取得した特徴量名の数 ({len(feature_names)}) と重要度の数 ({len(importances)}) が不一致です。インデックス名にフォールバックします。")
                    st.warning(f"取得された特徴量名サンプル: {feature_names[:15]}...") # サンプル数を増やす
                    feature_names = None # 不一致の場合はNoneに戻す
            else:
                 st.warning(f"特徴量名の取得に失敗しました。試行時のエラー: {err_msg}")


        except Exception as e_feat_outer:
            st.error(f"特徴量名の取得処理全体でエラー: {e_feat_outer}")
            traceback.print_exc()
            feature_names = None
        # ------------------------------------

        # 特徴量名が取得できなかった場合、連番を振る
        if feature_names is None:
             st.warning("最終的に特徴量名が確定できなかったため、インデックスを使用します。")
             feature_names = [f'feature_{i}' for i in range(len(importances))]

        # DataFrame作成
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        st.success("特徴量重要度データの抽出完了！")
        return importance_df

    except Exception as e:
        st.error(f"特徴量重要度データの抽出全体でエラーが発生しました: {e}")
        traceback.print_exc()
        return None 