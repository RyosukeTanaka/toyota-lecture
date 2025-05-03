# データ処理関連の関数
import pandas as pd
import streamlit as st
import os

@st.cache_data # Streamlitのキャッシュ機能を利用
def load_data(uploaded_file):
    """アップロードされたファイルからデータを読み込む"""
    try:
        uploaded_file.seek(0) # ファイルポインタをリセット
        df = pd.read_csv(uploaded_file)
        st.success("CSVファイルの読み込みに成功しました。")
        return df
    except pd.errors.EmptyDataError:
        st.error("エラー: アップロードされたCSVファイルが空です。")
        return None
    except Exception as e:
        st.error(f"CSVファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return None

def preprocess_data(df):
    """データの前処理（日付変換、価格差計算など）"""
    df_processed = df.copy()

    # --- 日付関連列の変換 ---
    # '利用日', '予約日' 列が存在するか確認し、存在すればdatetime型に変換
    # エラーが発生しても処理を続ける
    date_cols = ['利用日', '予約日']
    for col in date_cols:
        if col in df_processed.columns:
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except Exception as e:
                st.warning(f"警告: 列 '{col}' の日付変換中にエラーが発生しました: {e}")
        # else:
            # st.warning(f"警告: 日付変換対象の列 '{col}' がデータに含まれていません。")

    # --- リードタイムの再計算（日付変換後） ---
    # '利用日' と '予約日' が両方存在し、datetime型に変換成功していればリードタイムを日数で計算
    if '利用日' in df_processed.columns and '予約日' in df_processed.columns and \
       pd.api.types.is_datetime64_any_dtype(df_processed['利用日']) and \
       pd.api.types.is_datetime64_any_dtype(df_processed['予約日']):
        try:
            # 日付の差分を計算し、日数（整数）として取得
            df_processed['リードタイム_計算済'] = (df_processed['利用日'] - df_processed['予約日']).dt.days
            # 元のリードタイム列があれば、比較のために残しておくか、ここで削除するか選択
            # if 'リードタイム' in df_processed.columns:
            #     st.write("元の 'リードタイム' 列と計算結果 'リードタイム_計算済' を比較できます。")
        except Exception as e:
            st.warning(f"警告: リードタイムの計算中にエラーが発生しました: {e}")
            if 'リードタイム' in df_processed.columns:
                 df_processed['リードタイム_計算済'] = df_processed['リードタイム'] # 元の列を使う
            # else:
                 # st.warning("警告: リードタイムを計算できず、元の列も存在しません。")


    # --- 価格差特徴量の計算 ---
    # '価格_トヨタ', '価格_オリックス' 列が存在し、数値型であることを確認
    price_cols = ['価格_トヨタ', '価格_オリックス']
    if all(col in df_processed.columns for col in price_cols):
        # 数値型に変換試行 (変換できない値はNaNになる)
        for col in price_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # NaNが発生した場合の警告
        if df_processed[price_cols].isnull().any().any():
             st.warning(f"警告: '{', '.join(price_cols)}' 列に数値変換できない値、または欠損値が含まれています。価格差計算に影響する可能性があります。")

        # 価格差 (NaNを無視しないように計算)
        # df_processed['価格差'] = df_processed['価格_トヨタ'] - df_processed['価格_オリックス']
        # 価格比 (ゼロ除算を避ける, NaNを伝播)
        # df_processed['価格比'] = df_processed['価格_トヨタ'] / df_processed['価格_オリックス'].replace(0, pd.NA)

        # より頑健な計算（欠損値があっても計算できるように）
        df_processed['価格差'] = df_processed[price_cols[0]].sub(df_processed[price_cols[1]], fill_value=None) # fill_value=Noneで片方NaNなら結果もNaN
        # 価格比（オリックスが0またはNaNの場合はNaNにする）
        denominator = df_processed[price_cols[1]].replace(0, pd.NA)
        df_processed['価格比'] = df_processed[price_cols[0]].div(denominator, fill_value=None)

    else:
        st.warning(f"警告: 価格差計算に必要な列 ('{price_cols[0]}', '{price_cols[1]}') のいずれか、または両方がデータに含まれていません。")

    st.success("データの前処理（日付変換、リードタイム再計算、価格差計算）が完了しました。")
    return df_processed


def generate_exploration_report(df):
    """データ探索レポートをマークダウン形式で生成する"""
    markdown_report = []
    markdown_report.append("# データ探索レポート")
    markdown_report.append("\n---\n")

    # 基本情報
    shape_info = f"データ形状 (行数, 列数): {df.shape}"
    markdown_report.append("## 基本情報")
    markdown_report.append(shape_info)
    markdown_report.append("\n")

    # データ型
    dtypes_df = df.dtypes.reset_index().rename(columns={'index': '列名', 0: 'データ型'})
    markdown_report.append("## データ型")
    markdown_report.append(dtypes_df.to_markdown(index=False))
    markdown_report.append("\n")

    # 欠損値の数
    missing_values = df.isnull().sum()
    missing_df = missing_values[missing_values > 0].reset_index().rename(columns={'index': '列名', 0: '欠損値数'})
    markdown_report.append("## 欠損値の数")
    if not missing_df.empty:
        markdown_report.append(missing_df.to_markdown(index=False))
    else:
        markdown_report.append("欠損値はありません。")
    markdown_report.append("\n")

    # 数値データの基本統計量
    numeric_desc = df.describe(include=['number'])
    markdown_report.append("## 数値データの基本統計量")
    if not numeric_desc.empty:
        markdown_report.append(numeric_desc.to_markdown())
    else:
        markdown_report.append("数値データはありません。")
    markdown_report.append("\n")

    # カテゴリデータの基本統計量
    markdown_report.append("## カテゴリデータ（文字列など）の基本統計量")
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    if not categorical_cols.empty:
         cat_desc = df.describe(include=['object', 'category'])
         markdown_report.append(cat_desc.to_markdown())
    else:
         markdown_report.append("カテゴリデータはありません。")
    markdown_report.append("\n")

    return "\n".join(markdown_report)

def display_exploration(df):
    """Streamlit上でデータ探索結果を表示する"""
    st.header("データ探索")
    with st.expander("データセットの詳細を表示"):
        st.subheader("基本情報")
        st.write(f"データ形状 (行数, 列数): {df.shape}")

        st.subheader("データ型")
        dtypes_df = df.dtypes.reset_index().rename(columns={'index': '列名', 0: 'データ型'})
        # データ型列を文字列に変換してから表示
        dtypes_df['データ型'] = dtypes_df['データ型'].astype(str)
        st.dataframe(dtypes_df)

        st.subheader("欠損値の数")
        missing_values = df.isnull().sum()
        missing_df = missing_values[missing_values > 0].reset_index().rename(columns={'index': '列名', 0: '欠損値数'})
        if not missing_df.empty:
            # 念のため型を確認する場合があるが、通常このDFは問題ないはず
            st.dataframe(missing_df)
        else:
            st.write("欠損値はありません。")


        st.subheader("数値データの基本統計量")
        numeric_desc = df.describe(include=['number'])
        if not numeric_desc.empty:
             # describeの結果は通常数値なので問題ないことが多い
             st.dataframe(numeric_desc)
        else:
             st.write("数値データはありません。")


        st.subheader("カテゴリデータ（文字列など）の基本統計量")
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        if not categorical_cols.empty:
             cat_desc = df.describe(include=['object', 'category'])
             # describe(include='object') の結果は型が混在することがあるため、全体を文字列に変換
             st.dataframe(cat_desc.astype(str))
        else:
             st.write("カテゴリデータはありません。")

        # --- ダウンロードボタン --- #
        st.markdown("---") # 区切り線
        report_md = generate_exploration_report(df)
        st.download_button(
            label="📊 探索結果をダウンロード (.md)",
            data=report_md,
            file_name="data_exploration_report.md",
            mime="text/markdown",
        )


def filter_data_by_date(df, date_col, selected_date):
    """指定された日付列でデータをフィルタリングする"""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
         st.error(f"エラー: 列 '{date_col}' は日付型ではありません。前処理を確認してください。")
         return pd.DataFrame() # 空のDataFrameを返す
    try:
        # selected_dateもdatetime型に変換して比較
        selected_datetime = pd.to_datetime(selected_date).date() # 日付部分のみで比較
        return df[df[date_col].dt.date == selected_datetime]
    except Exception as e:
        st.error(f"日付によるフィルタリング中にエラーが発生しました: {e}")
        return pd.DataFrame()

def create_scenario_data(df_filtered, price_cols, scenario_type='mean', specific_prices=None):
    """価格変動なしシナリオのデータを作成する"""
    df_scenario = df_filtered.copy()

    if scenario_type == 'mean':
        # 分析対象期間（フィルター済みデータ）の平均価格を使用
        mean_prices = {}
        for col in price_cols:
            if col in df_scenario.columns and pd.api.types.is_numeric_dtype(df_scenario[col]):
                mean_prices[col] = df_scenario[col].mean()
            else:
                 st.warning(f"警告: シナリオ作成のため、列 '{col}' の平均値を計算できませんでした。")
                 return df_scenario # 平均が計算できない場合は元のデータを返す

        st.info(f"シナリオ作成: 期間中の平均価格 {mean_prices} を使用します。")
        for col, mean_val in mean_prices.items():
            df_scenario[col] = mean_val

    elif scenario_type == 'specific' and specific_prices:
        st.info(f"シナリオ作成: 指定された価格 {specific_prices} を使用します。")
        # 指定された価格を使用（例：予約開始時点の価格など）
        if len(price_cols) != len(specific_prices):
             st.error("エラー: シナリオ価格の数がprice_colsと一致しません。")
             return df_scenario # エラーの場合は元のデータを返す
        prices_to_set = dict(zip(price_cols, specific_prices))
        for col, price_val in prices_to_set.items():
            if col in df_scenario.columns:
                df_scenario[col] = price_val
            else:
                 st.warning(f"警告: シナリオ価格を設定する列 '{col}' がデータに存在しません。")

    else:
        st.error("エラー: 無効なシナリオタイプまたは価格が指定されていません。")
        return df_scenario

    # --- 価格差特徴量の再計算 ---
    # 再計算ロジックを再利用（関数化推奨）
    if all(col in df_scenario.columns for col in price_cols):
        # 数値型に変換試行 (シナリオデータなので基本数値のはずだが念のため)
        for col in price_cols:
            df_scenario[col] = pd.to_numeric(df_scenario[col], errors='coerce')

        # NaNが発生した場合の警告 (通常シナリオデータでは発生しないはず)
        if df_scenario[price_cols].isnull().any().any():
             st.warning(f"警告: シナリオデータの '{', '.join(price_cols)}' 列にNaNが含まれています。価格差再計算に影響する可能性があります。")

        # 価格差 (NaNを無視しないように計算)
        df_scenario['価格差'] = df_scenario[price_cols[0]].sub(df_scenario[price_cols[1]], fill_value=None)
        # 価格比 (ゼロ除算を避ける, NaNを伝播)
        denominator = df_scenario[price_cols[1]].replace(0, pd.NA)
        df_scenario['価格比'] = df_scenario[price_cols[0]].div(denominator, fill_value=None)
    # else:
        # price_colsが存在しないケースは上でハンドルされているはず

    return df_scenario 