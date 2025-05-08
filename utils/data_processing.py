# データ処理関連の関数
import pandas as pd
import streamlit as st
import os
import numpy as np # numpyをインポート
from typing import Optional, List, Dict, Tuple, Union # Dict, Tuple, Union を追加
from .constants import PRICE_COLUMNS # 定数を直接参照

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

# ★★★ 返り値の型ヒントと Docstring を変更 ★★★
def recalculate_lag_feature(df_processed: pd.DataFrame, lag_target_col: str, lag_days: int, booking_date_col: str, group_cols: list) -> Tuple[pd.DataFrame, Optional[Dict[str, Union[str, int]]]]:
    """指定された列のラグ特徴量を計算してDataFrameに追加（または上書き）し、計算結果情報（辞書）を返す"""
    lag_col_name = f'{lag_target_col}_lag{lag_days}'
    result_info = None # 計算結果情報用

    required_lag_cols = group_cols + [booking_date_col, lag_target_col]
    if not all(col in df_processed.columns for col in required_lag_cols):
        missing = [col for col in required_lag_cols if col not in df_processed.columns]
        st.warning(f"ラグ計算警告: 必要な列が見つかりません。不足列: {missing}")
        if lag_col_name not in df_processed.columns: df_processed[lag_col_name] = np.nan
        return df_processed, None # ★★★ エラー時は None を返す ★★★
    if not pd.api.types.is_datetime64_any_dtype(df_processed[booking_date_col]):
        st.warning(f"ラグ計算警告: '{booking_date_col}' が日付型ではありません。")
        if lag_col_name not in df_processed.columns: df_processed[lag_col_name] = np.nan
        return df_processed, None
    if not pd.api.types.is_numeric_dtype(df_processed[lag_target_col]):
         try:
              df_processed[lag_target_col] = pd.to_numeric(df_processed[lag_target_col], errors='coerce')
              if df_processed[lag_target_col].isnull().all(): raise ValueError("全てNaN")
         except Exception:
              st.warning(f"ラグ計算警告: '{lag_target_col}' が数値型に変換できません。")
              if lag_col_name not in df_processed.columns: df_processed[lag_col_name] = np.nan
              return df_processed, None

    try:
        df_temp = df_processed[required_lag_cols].sort_values(by=group_cols + [booking_date_col])
        df_temp['booking_date_minus_lag'] = df_temp[booking_date_col] - pd.Timedelta(days=lag_days)
        df_lookup = df_temp[group_cols + [booking_date_col, lag_target_col]].rename(
            columns={lag_target_col: lag_col_name, booking_date_col: 'lookup_booking_date'}
        )
        df_merged = pd.merge(
            df_temp, df_lookup,
            left_on=group_cols + ['booking_date_minus_lag'],
            right_on=group_cols + ['lookup_booking_date'],
            how='left', suffixes=('', '_lookup')
        )
        df_processed[lag_col_name] = df_merged.set_index(df_temp.index)[lag_col_name]

        nan_count = df_processed[lag_col_name].isnull().sum()
        result_info = {"lag_col_name": lag_col_name, "nan_count": nan_count}

    except Exception as e:
        st.error(f"ラグ特徴量 '{lag_col_name}' の計算中にエラーが発生しました: {e}")
        if lag_col_name not in df_processed.columns:
             df_processed[lag_col_name] = np.nan
        return df_processed, None # エラー時は None を返す

    return df_processed, result_info

def preprocess_data(df):
    """データの前処理（日付変換、価格差計算など） - ラグ計算は削除"""
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
    # price_cols = ['価格_トヨタ', '価格_オリックス'] # constantsから読み込むように変更

    if len(PRICE_COLUMNS) >= 2 and all(col in df_processed.columns for col in PRICE_COLUMNS[:2]): # 先頭2列で計算
        # 数値型に変換試行 (変換できない値はNaNになる)
        for col in PRICE_COLUMNS[:2]:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # NaNが発生した場合の警告
        if df_processed[PRICE_COLUMNS[:2]].isnull().any().any():
             st.warning(f"警告: '{PRICE_COLUMNS[0]}', '{PRICE_COLUMNS[1]}' 列に数値変換できない値、または欠損値が含まれています。価格差計算に影響する可能性があります。")

        # より頑健な計算（欠損値があっても計算できるように）
        df_processed['価格差'] = df_processed[PRICE_COLUMNS[0]].sub(df_processed[PRICE_COLUMNS[1]], fill_value=None) # fill_value=Noneで片方NaNなら結果もNaN
        # 価格比（オリックスが0またはNaNの場合はNaNにする）
        denominator = df_processed[PRICE_COLUMNS[1]].replace(0, pd.NA)
        df_processed['価格比'] = df_processed[PRICE_COLUMNS[0]].div(denominator, fill_value=None)

    elif len(PRICE_COLUMNS) == 1 and PRICE_COLUMNS[0] in df_processed.columns:
        st.info(f"価格関連特徴量: '{PRICE_COLUMNS[0]}' のみがPRICE_COLUMNSに設定されているため、価格差・価格比は計算されません。")
        # 必要であれば、価格差と価格比の列をNaNで作成
        if '価格差' not in df_processed.columns:
            df_processed['価格差'] = pd.NA
        if '価格比' not in df_processed.columns:
            df_processed['価格比'] = pd.NA
    else:
        st.warning(f"警告: 価格差計算に必要な列 ({ ', '.join(PRICE_COLUMNS) if PRICE_COLUMNS else 'なし'}) がデータに含まれていないか、PRICE_COLUMNSの設定が不十分です。価格差・価格比は計算されません。")
        if '価格差' not in df_processed.columns:
            df_processed['価格差'] = pd.NA
        if '価格比' not in df_processed.columns:
            df_processed['価格比'] = pd.NA

    # --- 価格比計算の後に追加 ---
    st.success("データの前処理（日付変換、リードタイム再計算、価格差計算）が完了しました。ラグ特徴量計算は別途実行されます。")

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

def find_last_price_change_lead_time(df_filtered, price_cols, lead_time_col):
    """
    指定された価格列について、価格が最後に変更されたリードタイムを返す。
    変更がない場合はNoneを返す。
    複数の価格列がある場合、最も小さいリードタイム（利用日に近い）を返す。
    価格変動が全くない価格列は、最終変更リードタイムの決定に影響しない。
    """
    last_change_lead_time = float('inf')
    # changed = False # changed_in_any_active_col に置き換え

    if df_filtered.empty or lead_time_col not in df_filtered.columns:
        st.warning("価格変更点検出: 入力データが空、またはリードタイム列が見つかりません。")
        return None

    # リードタイムでソート (降順: 大きい方から小さい方へ)
    df_sorted = df_filtered.sort_values(by=lead_time_col, ascending=False)

    active_price_cols = []
    for price_col in price_cols:
        if price_col not in df_sorted.columns:
            st.warning(f"価格変更点検出: 価格列 '{price_col}' が見つかりません。スキップします。")
            continue
        # dropna=True で NaN を無視してユニークな値の数を数える
        if df_sorted[price_col].nunique(dropna=True) > 1:
            active_price_cols.append(price_col)
        else:
            st.info(f"価格変更点検出: 価格列 '{price_col}' には価格変動がありませんでした。この列は分析から除外されます。")

    if not active_price_cols:
        st.info("価格最終変更点検出: 分析対象の価格列の中に、価格変動のあった列が見つかりませんでした。")
        return None

    changed_in_any_active_col = False # 初期化

    for price_col in active_price_cols: # 変動があった列のみをループ
        if price_col not in df_sorted.columns:
            # active_price_cols生成時にチェック済みだが念のため
            # st.warning(f"価格変更点検出: 価格列 '{price_col}' が見つかりません。")
            continue

        price_values = df_sorted[price_col]
        # 有効な（NaNでない）データポイントが2つ未満の場合、変動を検出しようがない
        if price_values.count() < 2:
            # st.info(f"価格列 '{price_col}' の有効なデータポイントが2未満のため、変動を検出できません。")
            continue

        # 現在の行の価格と、前のリードタイム（df_sortedでは1つ前の行）の価格を比較
        # df_sortedはリードタイム降順なので、shift(1)は時間的に「過去」のデータ（リードタイム大）
        shifted_prices = price_values.shift(1)
        price_changes = price_values.ne(shifted_prices)

        # 最初の行(最大のリードタイム)は常に変化なしとする (shift(1)の結果がNaNになるため比較結果に影響)
        if not price_changes.empty:
            price_changes.iloc[0] = False

        change_indices = price_changes[price_changes].index

        if not change_indices.empty:
            # 最もリードタイムが小さい変化点のリードタイムを取得
            current_col_last_change_lt = df_sorted.loc[change_indices, lead_time_col].min()
            last_change_lead_time = min(last_change_lead_time, current_col_last_change_lt)
            changed_in_any_active_col = True
        # else:
            # st.info(f"価格列 '{price_col}' では価格変動が見つかりませんでした。") # active_price_colsでフィルタ済みなので不要


    if not changed_in_any_active_col:
        # active_price_cols が空でなかったのにここに到達する場合、
        # 例えば、変動はあるが全てNaNになったなどの特殊ケースか、データが極端に少ない場合。
        st.info("価格最終変更点検出: 価格変動のある列は存在しましたが、具体的な変更点を特定できませんでした（データ不足の可能性など）。")
        return None
    elif last_change_lead_time == float('inf'):
         # changed_in_any_active_col が True なのに inf のまま -> ロジックエラーの可能性
         st.error("価格最終変更点検出: 予期せぬエラー。変動はあったがリードタイムが無限大のままです。")
         return None
    else:
        # st.info(f"価格最終変更リードタイム: {last_change_lead_time}")
        return int(last_change_lead_time) # 整数で返す


def create_scenario_data(df_filtered, price_cols, lead_time_col, scenario_type='mean', specific_prices=None, change_lead_time=None):
    """価格変動シナリオのデータを作成する"""
    df_scenario = df_filtered.copy()

    # --- シナリオ別の価格設定 ---
    if scenario_type == 'mean':
        # --- 平均価格シナリオ ---
        mean_prices = {}
        all_cols_found = True
        for col in price_cols:
            if col in df_scenario.columns and pd.api.types.is_numeric_dtype(df_scenario[col]):
                # NaNを除外して平均を計算
                mean_prices[col] = df_scenario[col].mean(skipna=True)
                if pd.isna(mean_prices[col]):
                    st.warning(f"警告: 平均価格計算で列 '{col}' が全てNaNでした。")
                    # この場合、価格は変更されない
            else:
                 st.warning(f"警告: シナリオ作成のため、列 '{col}' が見つからないか数値型ではありません。")
                 all_cols_found = False
                 # return pd.DataFrame() # エラーとして空を返すか、処理を続けるか

        if all_cols_found and mean_prices:
            st.info(f"シナリオ作成 (平均価格): 期間中の平均価格 { {k: round(v, 2) for k, v in mean_prices.items()} } を使用します。")
            for col, mean_val in mean_prices.items():
                if not pd.isna(mean_val): # 計算できた平均値のみ適用
                    df_scenario[col] = mean_val
        else:
            st.error("平均価格シナリオデータの作成に失敗しました。")
            return pd.DataFrame()


    elif scenario_type == 'last_change_fixed' and change_lead_time is not None:
        # --- 価格最終変更点以降固定シナリオ ---
        st.info(f"シナリオ作成 (最終変更点固定): 価格最終変更リードタイム ({change_lead_time}) 以降の価格を固定します。")

        # 基準となるリードタイム時点での価格を取得
        prices_at_change_time = {}
        base_data_rows = df_scenario[df_scenario[lead_time_col] == change_lead_time]

        if base_data_rows.empty:
             # 完全一致する行がない場合、そのリードタイム *以前* で最も近い行を探す
             available_lead_times_before = df_scenario[df_scenario[lead_time_col] <= change_lead_time][lead_time_col]
             if not available_lead_times_before.empty:
                 closest_lead_time = available_lead_times_before.max() # change_lead_time に最も近い（同じか小さい）LT
                 st.info(f"リードタイム {change_lead_time} の正確なデータ点がないため、最も近い過去のリードタイム {closest_lead_time} の価格を使用します。")
                 base_data_rows = df_scenario[df_scenario[lead_time_col] == closest_lead_time]
             else:
                 st.error(f"エラー: リードタイム {change_lead_time} またはそれ以前の価格データが見つかりません。")
                 return pd.DataFrame()

        if base_data_rows.empty:
             st.error(f"エラー: 基準となるリードタイム {change_lead_time} (またはそれ以前)のデータが見つかりません。")
             return pd.DataFrame()

        # 複数行ある場合も最初の行の価格を使う（ソートされている前提なら問題ないはず）
        base_row = base_data_rows.iloc[0]
        all_prices_found = True
        for col in price_cols:
            if col in base_row.index and not pd.isna(base_row[col]):
                prices_at_change_time[col] = base_row[col]
            else:
                 st.warning(f"警告: 固定価格取得のため、列 '{col}' がリードタイム {base_row[lead_time_col]} で見つからないか、値がNaNです。")
                 prices_at_change_time[col] = None # Noneを設定
                 all_prices_found = False # 一つでも価格が取れない場合はフラグを立てる

        # リードタイムが change_lead_time 以下の行の価格を固定価格で上書き
        target_period_mask = df_scenario[lead_time_col] <= change_lead_time
        st.write(f"リードタイム {change_lead_time} 以下の期間の価格を {prices_at_change_time} で固定します。")

        for col, fixed_price in prices_at_change_time.items():
            if fixed_price is not None and col in df_scenario.columns:
                df_scenario.loc[target_period_mask, col] = fixed_price
            elif col in df_scenario.columns:
                 # fixed_priceがNoneの場合 (上で警告済み)
                 # st.warning(f"警告: 列 '{col}' の固定価格が取得できなかったため、更新しません。")
                 pass


    elif scenario_type == 'specific' and specific_prices:
        # --- 特定価格指定シナリオ ---
        st.info(f"シナリオ作成 (指定価格): 指定された価格 {specific_prices} を使用します。")
        if len(price_cols) != len(specific_prices):
             st.error("エラー: シナリオ価格のリスト長がprice_colsと一致しません。")
             return pd.DataFrame()
        prices_to_set = dict(zip(price_cols, specific_prices))
        for col, price_val in prices_to_set.items():
            if col in df_scenario.columns:
                df_scenario[col] = price_val
            else:
                 st.warning(f"警告: シナリオ価格を設定する列 '{col}' がデータに存在しません。")


    else:
        # 'mean' 以外で、適切なパラメータがない場合
        if scenario_type != 'mean':
             st.error(f"エラー: 無効なシナリオタイプ({scenario_type})または必要なパラメータが不足しています。")
             return pd.DataFrame() # 空のDFを返す
        # 'mean' シナリオの場合は上で処理されている


    # --- 価格差特徴量の再計算 (共通処理) ---
    if all(col in df_scenario.columns for col in price_cols):
        # 再計算前に数値型であることを確認
        all_numeric = True
        for col in price_cols:
            try:
                df_scenario[col] = pd.to_numeric(df_scenario[col], errors='raise') # エラーがあればここで停止
            except Exception as e:
                st.error(f"価格差再計算エラー: 列 '{col}' を数値に変換できません: {e}")
                all_numeric = False
                break # ループ中断

        if all_numeric:
            # NaNチェック (数値変換後なので isna() でOK)
            if df_scenario[price_cols].isna().any().any():
                 st.warning(f"警告: シナリオデータの価格列 '{', '.join(price_cols)}' にNaNが含まれています。価格差/比の計算結果もNaNになる可能性があります。")

            try:
                df_scenario['価格差'] = df_scenario[price_cols[0]].sub(df_scenario[price_cols[1]], fill_value=np.nan) # fill_value=np.nan推奨
                # 価格比 (ゼロ除算とNaNを考慮)
                denominator = df_scenario[price_cols[1]].replace(0, np.nan) # 0をNaNに置換
                df_scenario['価格比'] = df_scenario[price_cols[0]].div(denominator) # これでNaN伝播とゼロ除算回避
            except Exception as e:
                st.error(f"価格差/比の再計算中にエラーが発生しました: {e}")
                # 必要に応じて '価格差', '価格比' 列を削除またはNaNで埋める
                if '価格差' in df_scenario.columns: df_scenario['価格差'] = np.nan
                if '価格比' in df_scenario.columns: df_scenario['価格比'] = np.nan

    # else:
        # 価格列が揃っていない場合は警告済みのはず

    st.success(f"シナリオデータ ({scenario_type}) 作成完了。")
    return df_scenario 