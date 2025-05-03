import streamlit as st
import pandas as pd
import os
# PyCaretの回帰モジュールをインポート
from pycaret.regression import setup, compare_models, pull, save_model

# ---定数---
# CSVファイルの相対パス (ワークスペースルートからのパス)
# 重要: この 'YOUR_CSV_FILE.csv' を実際のCSVファイル名に置き換えてください。
# CSV_FILE_PATH = 'YOUR_CSV_FILE.csv'

# ---関数---
# def load_data(file_path):
#     """指定されたパスからCSVファイルを読み込む"""
#     try:
#         # スクリプト(app.py)があるディレクトリを取得
#         script_dir = os.path.dirname(__file__)
#         # スクリプトのディレクトリとファイル名を結合して絶対パスを作成
#         abs_path = os.path.join(script_dir, file_path)
#
#         if not os.path.exists(abs_path):
#             st.error(f"エラー: ファイルが見つかりません。パスを確認してください: {abs_path}")
#             return None
#         df = pd.read_csv(abs_path)
#         return df
#     except FileNotFoundError:
#         # エラーメッセージも修正後のパスを表示するように念のため変更
#         script_dir = os.path.dirname(__file__)
#         abs_path = os.path.join(script_dir, file_path)
#         st.error(f"エラー: ファイルが見つかりません。パスを確認してください: {abs_path}")
#         return None
#     except pd.errors.EmptyDataError:
#         st.error("エラー: CSVファイルが空です。")
#         return None
#     except Exception as e:
#         st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
#         return None

# ---メイン処理---
st.title("利用台数予測システム") # タイトル変更

st.write("予測対象のCSVファイルをアップロードしてください:")

# ファイルアップローダーを設置
uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

if uploaded_file is not None:
    # アップロードされたファイルをデータフレームとして読み込む
    try:
        # アップロードされたファイルオブジェクトを直接渡す
        # ファイルポインタをリセットする必要があるかもしれないので、seek(0)を追加
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file)

        st.write("アップロードされたデータ（最初の20行）:")
        st.dataframe(data.head(20))

        st.markdown("---") # 区切り線

        # --- データ探索セクション ---
        st.header("データ探索")
        with st.expander("データセットの詳細を表示"):
            # マークダウンレポート生成用の文字列リスト
            markdown_report = [] # 初期化

            markdown_report.append("# データ探索レポート")
            markdown_report.append("\n---\n") # 区切り

            st.subheader("基本情報")
            shape_info = f"データ形状 (行数, 列数): {data.shape}"
            st.write(shape_info)
            markdown_report.append("## 基本情報")
            markdown_report.append(shape_info)
            markdown_report.append("\n")

            st.subheader("データ型")
            dtypes_df = data.dtypes.reset_index().rename(columns={'index': '列名', 0: 'データ型'})
            st.dataframe(dtypes_df)
            markdown_report.append("## データ型")
            markdown_report.append(dtypes_df.to_markdown(index=False))
            markdown_report.append("\n")

            st.subheader("欠損値の数")
            missing_values = data.isnull().sum()
            missing_df = missing_values[missing_values > 0].reset_index().rename(columns={'index': '列名', 0: '欠損値数'})
            markdown_report.append("## 欠損値の数")
            if not missing_df.empty:
                st.dataframe(missing_df)
                markdown_report.append(missing_df.to_markdown(index=False))
            else:
                no_missing_msg = "欠損値はありません。"
                st.write(no_missing_msg)
                markdown_report.append(no_missing_msg)
            markdown_report.append("\n")

            st.subheader("数値データの基本統計量")
            numeric_desc = data.describe(include=['number'])
            st.dataframe(numeric_desc)
            markdown_report.append("## 数値データの基本統計量")
            markdown_report.append(numeric_desc.to_markdown())
            markdown_report.append("\n")

            st.subheader("カテゴリデータ（文字列など）の基本統計量")
            markdown_report.append("## カテゴリデータ（文字列など）の基本統計量")
            # カテゴリデータが存在する場合のみ表示
            categorical_cols = data.select_dtypes(include=['object', 'category'])
            if not categorical_cols.empty:
                 cat_desc = data.describe(include=['object', 'category'])
                 st.dataframe(cat_desc)
                 markdown_report.append(cat_desc.to_markdown())
            else:
                 no_cat_msg = "カテゴリデータはありません。"
                 st.write(no_cat_msg)
                 markdown_report.append(no_cat_msg)
            markdown_report.append("\n")

            # --- ダウンロードボタン --- #
            st.markdown("---") # 区切り線
            # レポート文字列を結合
            final_markdown_report = "\n".join(markdown_report)
            st.download_button(
                label="📊 探索結果をダウンロード (.md)",
                data=final_markdown_report,
                file_name="data_exploration_report.md",
                mime="text/markdown",
            )
        # --- データ探索セクション終了 ---

        st.markdown("---") # 区切り線

        # PyCaret設定UI
        st.header("モデル設定")

        # 利用可能な列を取得
        available_columns = data.columns.tolist()

        # ターゲット変数の選択
        # デフォルトで '利用台数' が存在すればそれを、なければ最初の列を選択
        default_target = '利用台数' if '利用台数' in available_columns else available_columns[0]
        target_variable = st.selectbox(
            "予測するターゲット変数を選択してください:",
            available_columns,
            index=available_columns.index(default_target) # デフォルト選択
        )

        # 特徴量の選択 (ターゲット変数は除外)
        feature_options = [col for col in available_columns if col != target_variable]
        # デフォルトでは全ての数値・カテゴリ特徴量を選択候補とする（PyCaretに自動判別させるため、ここでは全選択をデフォルトにしない）
        selected_features = st.multiselect(
            "予測に使用する特徴量を選択してください:",
            feature_options,
            # default=feature_options # デフォルト全選択は一旦コメントアウト
        )

        # 評価モデルの選択
        # PyCaretの回帰で一般的に使われるモデルリスト（必要に応じて調整）
        available_models = ['lr', 'ridge', 'lasso', 'knn', 'dt', 'rf', 'et', 'lightgbm', 'xgboost', 'gbr', 'ada']
        selected_models = st.multiselect(
            "評価したいモデルを選択してください:",
            available_models,
            default=['lr', 'rf', 'lightgbm'] # デフォルトでいくつか選択
        )

        # 実行ボタン
        if st.button("モデル比較実行"):
            if not selected_features:
                st.warning("特徴量を1つ以上選択してください。")
            elif not selected_models:
                st.warning("評価するモデルを1つ以上選択してください。")
            else:
                st.info("PyCaretの処理を開始します。データ量によっては時間がかかる場合があります...")
                with st.spinner('モデルをセットアップし、比較中です...'):
                    try:
                        # PyCaretのセットアップ
                        # session_idを設定して再現性を確保
                        # numeric_featuresとcategorical_featuresは選択されたものだけ渡す
                        # setup関数は数値とカテゴリをある程度自動判別するが、明示的に指定する方が良い場合もある
                        numeric_features_selected = data[selected_features].select_dtypes(include=['number']).columns.tolist()
                        categorical_features_selected = data[selected_features].select_dtypes(exclude=['number']).columns.tolist()

                        s = setup(data,
                                  target=target_variable,
                                  numeric_features=numeric_features_selected if numeric_features_selected else None, # 空リストでエラーになる場合があるのでNoneに
                                  categorical_features=categorical_features_selected if categorical_features_selected else None,
                                  # use_gpu=True, # GPUがあればTrueにする（環境による）
                                  session_id=123, # 再現性のためのシード
                                  html=False)     # StreamlitではHTML出力をFalseに

                        st.success("セットアップ完了！")

                        # 選択されたモデルで比較
                        st.info(f"選択されたモデル: {', '.join(selected_models)} で比較を実行します...")
                        best_model = compare_models(include=selected_models,
                                                    fold=5, # クロスバリデーションの分割数
                                                    sort='RMSE') # RMSEでソート

                        # 結果の表示
                        st.success("モデル比較完了！")
                        st.header("モデル評価比較結果")
                        comparison_results = pull()
                        st.dataframe(comparison_results)

                        # (任意) 最良モデルの情報を表示
                        st.subheader("最も性能が良いモデル:")
                        st.write(best_model)


                    except Exception as e:
                        st.error(f"PyCaretの処理中にエラーが発生しました: {e}")
                        st.error("選択したターゲット変数、特徴量、データ型などを確認してください。")

    except pd.errors.EmptyDataError:
        st.error("エラー: アップロードされたCSVファイルが空です。")
    except Exception as e:
        st.error(f"CSVファイルの読み込みまたは処理中にエラーが発生しました: {e}")
else:
    st.info('CSVファイルをアップロードすると、ここにデータが表示され、予測モデルの比較ができます。') 