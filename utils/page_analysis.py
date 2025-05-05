# utils/page_analysis.py

import streamlit as st
import pandas as pd
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
    st.markdown("---")

    st.header(f"特定予約日以降の '{USAGE_COUNT_COLUMN}' 日別合計推移")
    st.write(f"指定した日付より**後**の予約日について、日ごとの '{USAGE_COUNT_COLUMN}' の合計値をグラフ表示します。")

    # --- 日別合計グラフ表示 & ゼロ日付特定 --- #
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
            nullify_result = None # Nullify結果用
            lag_recalc_result = None # Lag計算結果用

            with st.spinner("データ更新とラグ再計算を実行中..."):
                try:
                    current_data = st.session_state.get('processed_data')
                    if current_data is None:
                         update_message = "セッションからデータが見つかりません。"
                         # エラーメッセージは最後にまとめてサマリーに格納
                         st.error(f"エラー: {update_message}") # ここでは表示しておく
                         st.stop()

                    cutoff_date = pd.to_datetime(cutoff_date_str).date()
                    cols_to_null = [USAGE_COUNT_COLUMN, TARGET_VARIABLE]

                    # 1. データ更新
                    data_nulled, num_rows_updated, updated_cols = nullify_usage_data_after_date(
                        df=current_data.copy(),
                        cutoff_date=cutoff_date,
                        date_col=BOOKING_DATE_COLUMN,
                        cols_to_nullify=cols_to_null
                    )
                    # Nullify結果を保存
                    nullify_result = {"count": num_rows_updated, "cols": updated_cols}

                    if data_nulled is not None and num_rows_updated is not None:
                        # 2. ラグ特徴量再計算
                        # st.info("ラグ特徴量を再計算中...") # 冗長なのでコメントアウト
                        data_recalculated, lag_info = recalculate_lag_feature(
                            df_processed=data_nulled,
                            lag_target_col=LAG_TARGET_COLUMN,
                            lag_days=LAG_DAYS,
                            booking_date_col=BOOKING_DATE_COLUMN,
                            group_cols=LAG_GROUP_COLS
                        )
                        # Lag計算結果を保存
                        lag_recalc_result = lag_info

                        if data_recalculated is not None and isinstance(data_recalculated, pd.DataFrame):
                            st.session_state['processed_data'] = data_recalculated
                            update_status = "success"
                        else:
                            update_message = "ラグ特徴量の再計算に失敗しました。"
                    else:
                         update_message = "データ更新処理に失敗しました。"

                except Exception as e_update:
                    update_message = f"予期せぬエラー: {e_update}"
                    # エラー発生時もサマリーに情報を残す
                    st.error(f"データ更新またはラグ再計算中にエラーが発生しました: {update_message}")

                # --- 結果をセッションステートに保存 --- #
                st.session_state['last_update_summary'] = {
                    "status": update_status,
                    "message": update_message,
                    "date": cutoff_date_str,
                    "nullify_result": nullify_result,
                    "lag_recalc_result": lag_recalc_result
                }

                if 'zero_cutoff_date' in st.session_state:
                    del st.session_state['zero_cutoff_date']
                # st.rerun() は削除済み

            # ★★★ ボタン処理の最後に st.rerun() を追加 ★★★
            # これにより、スピナーが消え、ページ上部のサマリー表示が更新される
            st.rerun()

    else:
        st.info("先に上記の「日別合計推移」の分析を実行し、グラフが0になる日付を特定してください。") 