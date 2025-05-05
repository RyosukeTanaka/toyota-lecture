# utils/page_analysis.py

import streamlit as st
import pandas as pd
import datetime # 保存ファイル名用に追加
from .constants import ( # constants からインポート
    USAGE_COUNT_COLUMN, TARGET_VARIABLE, BOOKING_DATE_COLUMN,
    LAG_TARGET_COLUMN, LAG_DAYS, LAG_GROUP_COLS
)
from .data_processing import ( # 相対インポートに変更
    display_exploration,
    recalculate_lag_feature # ラグ再計算関数
)
from .ui_components import ( # 相対インポートに変更
    render_data_analysis_sidebar_widgets
)
from .analysis import analyze_daily_sum_after_date # 相対インポートに変更
from .data_modification import nullify_usage_data_after_date # 相対インポートに変更

# --- データ分析・修正ページ描画関数 ---
def render_data_analysis_page(data: pd.DataFrame):
    st.title("データ分析・修正")

    # --- 前回の更新サマリー表示 --- #
    if 'last_update_summary' in st.session_state and st.session_state['last_update_summary']:
        summary = st.session_state['last_update_summary']
        st.markdown("---")
        st.subheader("前回のデータ更新・再計算結果")
        if summary.get("status") == "success":
            # ★★★ Nullify と Lag の両方の結果を表示 ★★★
            nullify_res = summary.get("nullify_result", {})
            lag_res = summary.get("lag_recalc_result", {})

            success_messages = []
            if nullify_res.get("count") is not None: # count が 0 でも表示
                 if nullify_res.get("count", 0) > 0:
                     success_messages.append(
                         f"**{summary.get('date', '不明')}**以降の**{nullify_res.get('count', 0)}行**で列**{nullify_res.get('cols', [])}**をNAに更新。"
                     )
                 else:
                     success_messages.append(
                         f"**{summary.get('date', '不明')}**以降のデータ更新対象なし。"
                     )

            if lag_res:
                 success_messages.append(
                     f"ラグ特徴量**'{lag_res.get('lag_col_name', '?')}'**を再計算({lag_res.get('nan_count', '?')}行NaN)。"
                 )

            if success_messages:
                 st.success("✅ " + " ".join(success_messages))
            else:
                 st.info("ℹ️ データ更新・再計算処理は実行されましたが、変更はありませんでした。")

        else: # status が 'error' または不明な場合
             st.warning(f"⚠️ 前回の処理で問題が発生: {summary.get('message', '不明なエラー')}")

        del st.session_state['last_update_summary']
        st.markdown("---")

    # --- サイドバーウィジェット --- #
    selected_analysis_date, analyze_button = render_data_analysis_sidebar_widgets(data)

    # --- メインエリア --- #
    st.header("データ探索")
    display_exploration(data)

    # --- 日別合計グラフ表示 & ゼロ日付特定 --- #
    st.header(f"特定予約日以降の '{USAGE_COUNT_COLUMN}' 日別合計推移")
    st.write(f"指定した日付より**後**の予約日について、日ごとの '{USAGE_COUNT_COLUMN}' の合計値をグラフ表示します。")
    if selected_analysis_date is not None:
        daily_sum_df = analyze_daily_sum_after_date(data=data, start_date=selected_analysis_date, booking_date_col=BOOKING_DATE_COLUMN, sum_col=USAGE_COUNT_COLUMN)
        if daily_sum_df is not None:
            if not daily_sum_df.empty:
                st.line_chart(daily_sum_df)
                zero_date_str = None
                daily_sum_df_sorted = daily_sum_df.sort_index()
                sum_col_name = f'{USAGE_COUNT_COLUMN}_合計'
                if sum_col_name in daily_sum_df_sorted.columns:
                     zero_rows = daily_sum_df_sorted[daily_sum_df_sorted[sum_col_name] <= 0]
                     if not zero_rows.empty:
                         zero_date_str = zero_rows.index[0].strftime('%Y-%m-%d')
                         st.info(f"📈グラフの値が最初に0になった日付: **{zero_date_str}**")
                         st.session_state['zero_cutoff_date'] = zero_date_str
                     else:
                         st.info("📈グラフ期間中に値が0になる日はありませんでした。")
                         if 'zero_cutoff_date' in st.session_state: del st.session_state['zero_cutoff_date']
                else: st.warning("グラフデータから合計列が見つかりませんでした。")
                with st.expander("詳細データ表示"):
                     st.dataframe(daily_sum_df)
    else:
        st.info("サイドバーで起点となる予約日を選択してください。")

    # --- データ更新とラグ再計算セクション --- #
    st.markdown("---")
    st.header("データ更新とラグ特徴量再計算")
    if 'zero_cutoff_date' in st.session_state and st.session_state['zero_cutoff_date']:
        cutoff_date_str = st.session_state['zero_cutoff_date']
        st.write(f"特定された日付 **{cutoff_date_str}** 以降のデータについて、")
        st.write(f"'{USAGE_COUNT_COLUMN}' および '{TARGET_VARIABLE}' を欠損値(NA)に更新し、")
        st.write(f"'{LAG_TARGET_COLUMN}_lag{LAG_DAYS}' を再計算します。")
        update_button = st.button(f"🔄 {cutoff_date_str} 以降のデータを更新＆ラグ再計算", key="update_data_button")
        if update_button:
            update_status = "error"
            update_message = ""
            nullify_result = None
            lag_recalc_result = None
            with st.spinner("データ更新とラグ再計算を実行中..."):
                try:
                    current_data = st.session_state.get('processed_data')
                    if current_data is None:
                         update_message = "セッションからデータが見つかりません。"
                         st.error(f"エラー: {update_message}")
                         st.stop()
                    cutoff_date = pd.to_datetime(cutoff_date_str).date()
                    cols_to_null = [USAGE_COUNT_COLUMN, TARGET_VARIABLE]
                    data_nulled, num_rows_updated, updated_cols = nullify_usage_data_after_date(
                        df=current_data.copy(), cutoff_date=cutoff_date, date_col=BOOKING_DATE_COLUMN, cols_to_nullify=cols_to_null
                    )
                    nullify_result = {"count": num_rows_updated, "cols": updated_cols}
                    if data_nulled is not None and num_rows_updated is not None:
                        data_recalculated, lag_info = recalculate_lag_feature(
                            df_processed=data_nulled, lag_target_col=LAG_TARGET_COLUMN, lag_days=LAG_DAYS,
                            booking_date_col=BOOKING_DATE_COLUMN, group_cols=LAG_GROUP_COLS
                        )
                        lag_recalc_result = lag_info
                        if data_recalculated is not None and isinstance(data_recalculated, pd.DataFrame):
                            st.session_state['processed_data'] = data_recalculated
                            update_status = "success"
                        else: update_message = "ラグ特徴量の再計算に失敗しました。"
                    else: update_message = "データ更新処理に失敗しました。"
                except Exception as e_update: update_message = f"予期せぬエラー: {e_update}"
                st.session_state['last_update_summary'] = {
                    "status": update_status, "message": update_message, "date": cutoff_date_str,
                    "nullify_result": nullify_result, "lag_recalc_result": lag_recalc_result
                }
                if 'zero_cutoff_date' in st.session_state: del st.session_state['zero_cutoff_date']
            st.rerun()
    else:
        st.info("先に上記の「日別合計推移」の分析を実行し、グラフが0になる日付を特定してください。")

    # --- 列削除セクション --- #
    st.markdown("---")
    st.header("列の削除")
    st.write("不要な列を選択して削除できます。")
    current_data_cols = data.columns.tolist()
    cols_to_delete = st.multiselect(
        "削除する列を選択してください:",
        options=current_data_cols,
        key="delete_columns_multiselect"
    )
    delete_cols_button = st.button("🗑️ 選択した列を削除", key="delete_columns_button")
    if delete_cols_button and cols_to_delete:
        with st.spinner("列を削除中..."):
            try:
                current_data = st.session_state.get('processed_data')
                if current_data is not None:
                    data_after_delete = current_data.drop(columns=cols_to_delete)
                    st.session_state['processed_data'] = data_after_delete
                    st.success(f"列 {cols_to_delete} を削除しました。")
                    st.rerun()
                else:
                    st.error("セッションからデータが見つかりません。")
            except KeyError as e:
                 st.error(f"列の削除中にエラーが発生しました: 存在しない列 {e} が指定された可能性があります。")
            except Exception as e:
                 st.error(f"列の削除中に予期せぬエラーが発生しました: {e}")
    elif delete_cols_button and not cols_to_delete:
         st.warning("削除する列が選択されていません。")

    # ★★★ ここに移動 ★★★
    st.markdown("---")
    st.header("行の削除（欠損値基準）")
    st.write("指定した列に欠損値(NaN)が含まれる行を削除します。")
    current_data_cols_for_nan_check = data.columns.tolist()
    cols_to_check_for_nan = st.multiselect(
        "欠損値(NaN)をチェックする列を選択してください:",
        options=current_data_cols_for_nan_check,
        key="nan_check_columns_multiselect"
    )
    delete_nan_rows_button = st.button("🧹 選択列のNaN行を削除", key="delete_nan_rows_button")
    if delete_nan_rows_button and cols_to_check_for_nan:
        with st.spinner("NaNを含む行を削除中..."):
            try:
                current_data_nan = st.session_state.get('processed_data')
                if current_data_nan is not None:
                    rows_before_drop = len(current_data_nan)
                    nan_rows_mask = current_data_nan[cols_to_check_for_nan].isnull().any(axis=1)
                    deleted_nan_rows = None
                    if nan_rows_mask.any():
                        deleted_nan_rows = current_data_nan[nan_rows_mask].copy()
                    data_after_drop_nan = current_data_nan.dropna(subset=cols_to_check_for_nan)
                    rows_after_drop = len(data_after_drop_nan)
                    rows_removed = rows_before_drop - rows_after_drop
                    st.session_state['processed_data'] = data_after_drop_nan
                    st.success(f"列 {cols_to_check_for_nan} のいずれかにNaNを含む {rows_removed} 行を削除しました。")
                    if deleted_nan_rows is not None and not deleted_nan_rows.empty:
                         with st.expander(f"削除された {rows_removed} 行の詳細を表示"):
                              st.dataframe(deleted_nan_rows)
                    # st.rerun() # コメントアウト継続
                else:
                    st.error("セッションからデータが見つかりません。")
            except KeyError as e:
                 st.error(f"NaN行の削除中にエラーが発生しました: 存在しない列 {e} が指定された可能性があります。")
            except Exception as e:
                 st.error(f"NaN行の削除中に予期せぬエラーが発生しました: {e}")
    elif delete_nan_rows_button and not cols_to_check_for_nan:
         st.warning("NaNをチェックする列が選択されていません。")

    # --- 修正済みデータの保存セクション --- #
    st.markdown("---")
    st.header("修正済みデータの保存")
    st.write("現在のデータフレームの状態（データ更新、列削除など）をCSVファイルとして保存します。")

    # 現在のデータを取得
    current_data_to_save = st.session_state.get('processed_data')

    if current_data_to_save is not None and isinstance(current_data_to_save, pd.DataFrame):
        try:
            csv_data = current_data_to_save.to_csv(index=False).encode('utf-8')
            # ファイル名を生成（元のファイル名があれば使う、なければタイムスタンプ）
            original_filename = st.session_state.get('last_uploaded_filename', 'data')
            if original_filename.endswith('.csv'):
                original_filename_base = original_filename[:-4]
            else:
                original_filename_base = original_filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"{original_filename_base}_modified_{timestamp}.csv"

            st.download_button(
                label="💾 修正済みデータをCSVとして保存",
                data=csv_data,
                file_name=download_filename,
                mime='text/csv',
                key='download_modified_data_button'
            )
        except Exception as e:
            st.error(f"CSVデータへの変換またはダウンロードボタンの生成中にエラーが発生しました: {e}")
    elif current_data_to_save is None:
         st.warning("保存対象のデータがありません。ファイルをアップロードしてください。")
    else:
         st.warning("セッションデータが予期せぬ形式です。") 