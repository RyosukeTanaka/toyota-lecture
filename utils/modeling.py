# 機械学習モデル関連の関数
from pycaret.regression import setup, compare_models, predict_model, pull
import streamlit as st
import pandas as pd

# グローバル変数やセッションステートでモデルを保持することも検討できるが、
# ここでは関数間でモデルオブジェクトを渡すシンプルな方法を採用。

@st.cache_resource # モデルオブジェクトのような大きなリソースは st.cache_resource を使う
def setup_and_compare_models(_data, target, numeric_features, categorical_features, include_models, sort_metric='RMSE', fold=5):
    """PyCaretのセットアップとモデル比較を実行し、最良モデルと結果を返す"""
    st.info("PyCaretのセットアップを開始します...")
    try:
        # html=FalseにしないとStreamlit内で表示がおかしくなることがある
        s = setup(data=_data, target=target,
                  numeric_features=numeric_features if numeric_features else None,
                  categorical_features=categorical_features if categorical_features else None,
                  session_id=123, # 再現性のため
                  verbose=False, # Streamlit上では詳細ログを抑制
                  html=False)
        st.success("PyCaret セットアップ完了！")

        st.info(f"選択されたモデル: {', '.join(include_models)} で比較を実行します (fold={fold}, sort='{sort_metric}')...")
        # _ fica warning
        best_model = compare_models(include=include_models, fold=fold, sort=sort_metric,
                                    verbose=False) # 詳細ログ抑制
        comparison_results = pull()
        st.success("モデル比較完了！")
        st.write("最良モデル:", best_model)
        return best_model, comparison_results

    except Exception as e:
        st.error(f"PyCaret処理中にエラーが発生しました: {e}")
        st.error("ターゲット変数、特徴量の選択、データ型などを確認してください。")
        # traceback.print_exc() # 詳細なトレースバックが必要な場合
        return None, pd.DataFrame() # エラー時はNoneと空のDataFrameを返す


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