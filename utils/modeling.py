# 機械学習モデル関連の関数
from pycaret.regression import setup, compare_models, predict_model, pull, plot_model
import streamlit as st
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import numpy as np # numpyをインポート

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


def predict_with_model(model, _data, target: str):
    """与えられたモデル、データ、ターゲット列名で予測を実行する"""
    if model is None or _data.empty:
        st.error("予測に必要なモデルまたはデータがありません。")
        return pd.DataFrame() # 空のDataFrameを返す
    try:
        # --- 予測データからターゲット列を削除 ---
        data_for_prediction = _data.copy()
        if target in data_for_prediction.columns:
            data_for_prediction = data_for_prediction.drop(columns=[target])
            st.info(f"予測実行のため、データから列 '{target}' を一時的に削除しました。")
        # else: # 警告は削除しても良い
        #    st.warning(f"予測データにターゲット列 '{target}' が見つかりません。")

        # ★★★ 追加: 予測データの特徴量に含まれるNaNを強制的に補完 ★★★
        st.info("予測データの特徴量のNaNを補完しています...")
        original_nan_counts = data_for_prediction.isnull().sum()
        cols_with_nan = original_nan_counts[original_nan_counts > 0].index.tolist()

        if cols_with_nan:
            st.warning(f"予測データの特徴量でNaNが見つかりました。補完を実行します: {cols_with_nan}")
            for col in cols_with_nan:
                if pd.api.types.is_numeric_dtype(data_for_prediction[col]):
                    # 数値列は平均値で補完
                    mean_val = data_for_prediction[col].mean()
                    data_for_prediction[col] = data_for_prediction[col].fillna(mean_val)
                    st.write(f"- 数値列 '{col}' のNaNを平均値 ({mean_val:.2f}) で補完しました。")
                elif pd.api.types.is_object_dtype(data_for_prediction[col]) or pd.api.types.is_categorical_dtype(data_for_prediction[col]):
                    # カテゴリ列は最頻値で補完
                    mode_val = data_for_prediction[col].mode()
                    if not mode_val.empty:
                        fill_value = mode_val[0]
                        data_for_prediction[col] = data_for_prediction[col].fillna(fill_value)
                        st.write(f"- カテゴリ列 '{col}' のNaNを最頻値 ({fill_value}) で補完しました。")
                    else:
                        # 最頻値がない場合 (すべてNaNなど) は 'missing' で埋める
                        data_for_prediction[col] = data_for_prediction[col].fillna('missing')
                        st.write(f"- カテゴリ列 '{col}' は最頻値がなかったため 'missing' で補完しました。")
                else:
                     # その他の型はとりあえず 'missing' で埋める (またはエラーにする)
                     data_for_prediction[col] = data_for_prediction[col].fillna('missing_unknown_type')
                     st.warning(f"- 列 '{col}' は未知の型のため 'missing_unknown_type' で補完しました。")

            # 補完後のNaN確認 (デバッグ用)
            final_nan_counts = data_for_prediction.isnull().sum().sum()
            if final_nan_counts == 0:
                 st.success("予測データの特徴量のNaN補完が完了しました。")
            else:
                 st.error(f"NaNの補完後も {final_nan_counts} 個のNaNが残っています。処理を確認してください。")
                 st.dataframe(data_for_prediction[data_for_prediction.isnull().any(axis=1)]) # NaNが残っている行を表示
                 return pd.DataFrame() # エラーとして処理中断
        else:
             st.info("予測データの特徴量にNaNはありませんでした。")
        # ★★★ ----------------------------------------------- ★★★


        st.info("予測を実行しています...")
        # predict_model に渡すデータを変更
        predictions_result = predict_model(model, data=data_for_prediction)
        st.success("予測完了！")

        # ★★★ 結果を元のデータフレーム (_data) と結合して返す ★★★
        # predict_model は入力データ (ターゲット列なし) に prediction_label を追加して返す
        # 元の _data のインデックスと結合して、他の列情報 (NaNのターゲット含む) を復元
        predictions_final = _data.copy()
        # predict_model が返す DF の prediction_label 列を結合
        # インデックスが一致していることを確認
        if predictions_final.index.equals(predictions_result.index):
             predictions_final['prediction_label'] = predictions_result['prediction_label']
        else:
             # インデックスが異なる場合、より安全なマージを試みる (元のデータのインデックスを優先)
             st.warning("予測結果と元のデータのインデックスが一致しません。インデックスに基づいてマージします。")
             predictions_final = pd.merge(
                 predictions_final,
                 predictions_result[['prediction_label']], # 予測値のみ取得
                 left_index=True,
                 right_index=True,
                 how='left' # 元のデータに予測値を結合
             )
             if predictions_final['prediction_label'].isnull().any():
                  st.warning("予測結果のマージ後、一部の行で予測値がNaNになりました。インデックス不一致の可能性があります。")

        return predictions_final

    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc()) # 詳細なトレースバックを表示
        return pd.DataFrame()

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