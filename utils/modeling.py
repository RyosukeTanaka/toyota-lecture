# 機械学習モデル関連の関数
from pycaret.regression import setup, compare_models, predict_model, pull, plot_model
import streamlit as st
import pandas as pd
import traceback
import matplotlib.pyplot as plt

# グローバル変数やセッションステートでモデルを保持することも検討できるが、
# ここでは関数間でモデルオブジェクトを渡すシンプルな方法を採用。

# @st.cache_resource # 一時的にキャッシュを無効化
def setup_and_compare_models(_data, target, numeric_features, categorical_features, include_models, sort_metric='RMSE', fold=5):
    """PyCaretのセットアップとモデル比較を実行し、最良モデル、結果、セットアップオブジェクトを返す"""
    st.info("PyCaretのセットアップを開始します...")
    setup_result = None # 初期化
    try:
        # html=FalseにしないとStreamlit内で表示がおかしくなることがある
        setup_result = setup(data=_data, target=target,
                       numeric_features=numeric_features if numeric_features else None,
                       categorical_features=categorical_features if categorical_features else None,
                       session_id=123, # 再現性のため
                       verbose=False, # Streamlit上では詳細ログを抑制
                       html=False)
        st.success("PyCaret セットアップ完了！")

        st.info(f"選択されたモデル: {', '.join(include_models)} で比較を実行します (fold={fold}, sort='{sort_metric}')...")
        best_model = compare_models(include=include_models, fold=fold, sort=sort_metric,
                                    verbose=False)
        comparison_results = pull()
        st.success("モデル比較完了！")
        st.write("最良モデル:", best_model)
        # セットアップ結果(setup_result)も返すように変更
        return best_model, comparison_results, setup_result

    except Exception as e:
        st.error(f"PyCaret処理中にエラーが発生しました: {e}")
        st.error("ターゲット変数、特徴量の選択、データ型などを確認してください。")
        # traceback.print_exc()
        # セットアップ結果も含めてNoneや空のDataFrameを返す
        return None, pd.DataFrame(), setup_result


def predict_with_model(model, _data):
    """与えられたモデルとデータで予測を実行する"""
    if model is None or _data.empty:
        st.error("予測に必要なモデルまたはデータがありません。")
        return pd.DataFrame() # 空のDataFrameを返す
    try:
        st.info("予測を実行しています...")
        predictions = predict_model(model, data=_data)
        st.success("予測完了！")
        # predict_modelは元のデータに 'prediction_label' 列を追加して返す
        return predictions
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")
        # traceback.print_exc()
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
    """訓練済みモデルとPyCaretのセットアップ結果から特徴量重要度を抽出し、DataFrameで返す"""
    if model is None or setup_result is None:
        st.warning("重要度抽出: モデルまたはセットアップ結果がありません。")
        return None

    try:
        st.info("特徴量重要度データを抽出中...")

        # --- 最終的なモデル推定器を取得 --- #
        if hasattr(model, 'steps'):
            final_estimator = model.steps[-1][1]
        else:
            st.warning("モデルがPipelineではないようです。そのままモデルを使用します。")
            final_estimator = model

        # --- 重要度を取得 --- #
        if hasattr(final_estimator, 'feature_importances_'):
            importances = final_estimator.feature_importances_
        else:
            st.error(f"モデル ({type(final_estimator).__name__}) から feature_importances_ 属性が見つかりません。")
            return None

        # --- 前処理後の特徴量名を取得 (修正箇所) --- #
        feature_names = None
        try:
            # 優先度1: setup_resultオブジェクトの get_config メソッドを使用 (PyCaret 3.x 推奨)
            if hasattr(setup_result, 'get_config'):
                # 'X_transformed' や 'pipeline' から特徴量名を取得できないか試す
                # setup_result.get_config('X_transformed') はデータそのもの
                # setup_result.pipeline でパイプラインを取得
                pipeline_ = setup_result.pipeline
                if hasattr(pipeline_, 'get_feature_names_out'):
                    # モデルを除いたパイプラインで名前取得
                    if hasattr(model, 'steps'):
                        feature_names = list(model[:-1].get_feature_names_out())
                    else:
                         # モデルがパイプラインでない場合、前処理がないと仮定？
                         # または setup_result.pipeline 全体で取得？
                         # feature_names = list(pipeline_.get_feature_names_out())
                         st.warning("モデルがPipelineでないため、特徴量名の取得方法が不明です。")
                         pass # feature_names は None のまま
                # get_config に直接特徴量リストがある場合も？ (例: 'feature_names_transformed')
                elif setup_result.get_config('feature_names_transformed') is not None:
                    feature_names = setup_result.get_config('feature_names_transformed')
                    st.info("get_config('feature_names_transformed') から特徴量名を取得しました。")
                elif setup_result.get_config('X_train_transformed') is not None:
                     # fallback: 変換後の訓練データのカラム名を使う
                     feature_names = setup_result.get_config('X_train_transformed').columns.tolist()
                     st.info("get_config('X_train_transformed').columns から特徴量名を取得しました。")

            # 優先度2: setup_resultオブジェクトの属性を直接参照 (旧バージョンなど)
            elif hasattr(setup_result, 'feature_names_transformed'):
                feature_names = setup_result.feature_names_transformed
                st.info("setup_result.feature_names_transformed 属性から特徴量名を取得しました。")
            elif hasattr(setup_result, 'X_train_transformed'):
                 feature_names = setup_result.X_train_transformed.columns.tolist()
                 st.info("setup_result.X_train_transformed.columns 属性から特徴量名を取得しました。")
            else:
                 st.warning("setup_result から特徴量名を取得する既知の方法が見つかりませんでした。")

            # --- 長さチェックと調整 --- #
            if feature_names is not None:
                if len(feature_names) == len(importances):
                    st.success("特徴量名と重要度の数が一致しました。")
                else:
                    st.warning(f"取得した特徴量名の数 ({len(feature_names)}) と重要度の数 ({len(importances)}) が不一致です。インデックス名にフォールバックします。")
                    st.warning(f"取得された特徴量名サンプル: {feature_names[:5]}...") # デバッグ用
                    feature_names = None # 不一致の場合はNoneに戻す

        except Exception as e_feat:
            st.error(f"特徴量名の取得中に予期せぬエラーが発生しました: {e_feat}")
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