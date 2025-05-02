import streamlit as st
import pandas as pd
import os

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
st.title("CSVデータビューワー")

st.write("表示したいCSVファイルをアップロードしてください:")

# ファイルアップローダーを設置
uploaded_file = st.file_uploader("CSVファイルを選択", type='csv')

if uploaded_file is not None:
    # アップロードされたファイルをデータフレームとして読み込む
    try:
        # StringIOを使わずに直接アップロードされたファイルオブジェクトを渡せる
        data = pd.read_csv(uploaded_file)

        st.write("データの最初の20行:")
        # データフレームの先頭20行を表示
        st.dataframe(data.head(20))

        # 必要であれば全データも表示 (コメントアウトされています)
        # st.write("---")
        # st.write("全てのデータ:")
        # st.dataframe(data)

    except pd.errors.EmptyDataError:
        st.error("エラー: アップロードされたCSVファイルが空です。")
    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
else:
    st.info('CSVファイルをアップロードすると、ここにデータが表示されます。') 