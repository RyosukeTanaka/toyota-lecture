import streamlit as st
import pandas as pd
import datetime

from .constants import DATE_COLUMN, CAR_CLASS_COLUMN, LEAD_TIME_COLUMN, PRICE_COLUMNS
from .analysis import analyze_price_change_details_in_range
from .visualization import plot_price_change_lead_time_distribution, plot_price_change_magnitude_scatter

def render_price_analysis_page(data: pd.DataFrame):
    st.title("価格変動分析")
    st.markdown("""
    指定した**利用日**の範囲内で、**価格_トヨタ**の価格が変更されたリードタイム、
    その変更前後の価格、および関連する詳細情報をテーブルとグラフで表示します。
    価格変更のタイミングや傾向を把握するのに役立ちます。
    """)

    if data is None or data.empty:
        st.warning("分析対象のデータがありません。CSVファイルをアップロードしてください。")
        return

    # データプレビュー (オプション)
    with st.expander("データプレビュー (先頭5行)"):
        st.dataframe(data.head())

    st.markdown("---")

    # 日付範囲選択
    # current_data_for_price_analysis = st.session_state.get('processed_data', data) # processed_dataはここでは使わない想定
    current_data_for_price_analysis = data 
    min_analysis_date = None
    max_analysis_date = None
    if DATE_COLUMN in current_data_for_price_analysis.columns and pd.api.types.is_datetime64_any_dtype(current_data_for_price_analysis[DATE_COLUMN]):
        valid_dates_for_price = current_data_for_price_analysis[DATE_COLUMN].dropna().dt.date
        if not valid_dates_for_price.empty:
            min_analysis_date = valid_dates_for_price.min()
            max_analysis_date = valid_dates_for_price.max()
    
    if min_analysis_date and max_analysis_date:
        col_pa1, col_pa2 = st.columns(2)
        
        default_start_date = datetime.date(2025, 4, 1)
        default_end_date = datetime.date(2025, 4, 14)

        if default_start_date < min_analysis_date:
            actual_default_start = min_analysis_date
        elif default_start_date > max_analysis_date:
            actual_default_start = max_analysis_date
        else:
            actual_default_start = default_start_date

        if default_end_date > max_analysis_date:
            actual_default_end = max_analysis_date
        elif default_end_date < actual_default_start:
            actual_default_end = actual_default_start
        else:
            actual_default_end = default_end_date
            
        with col_pa1:
            price_analysis_start_date = st.date_input(
                "分析開始日（利用日）:", 
                value=actual_default_start, 
                min_value=min_analysis_date, 
                max_value=max_analysis_date, 
                key="price_analysis_start_page"
            )
        with col_pa2:
            price_analysis_end_date = st.date_input(
                "分析終了日（利用日）:", 
                value=actual_default_end, 
                min_value=price_analysis_start_date, 
                max_value=max_analysis_date, 
                key="price_analysis_end_page"
            )
        
        analyze_price_changes_button = st.button("価格変動点を分析", key="analyze_price_changes_page")

        if analyze_price_changes_button:
            with st.spinner("価格変動点を分析中..."):
                target_price_col_for_analysis = [PRICE_COLUMNS[0]] if PRICE_COLUMNS else []
                if not target_price_col_for_analysis:
                    st.error("分析対象の価格列（例: 価格_トヨタ）が定数ファイルで定義されていません。")
                    st.stop()
                    
                price_change_df = analyze_price_change_details_in_range(
                    data=current_data_for_price_analysis,
                    start_date=price_analysis_start_date,
                    end_date=price_analysis_end_date,
                    date_col=DATE_COLUMN,
                    car_class_col=CAR_CLASS_COLUMN,
                    lead_time_col=LEAD_TIME_COLUMN,
                    price_cols=target_price_col_for_analysis 
                )
            
            if not price_change_df.empty:
                st.subheader("価格変動点 詳細")
                st.dataframe(
                    price_change_df.style.format({
                        "変更前価格": "{:,.0f}", # 小数点以下なし、カンマ区切り
                        "変更後価格": "{:,.0f}", # 小数点以下なし、カンマ区切り
                        # 必要に応じて他の数値列もフォーマット指定可能
                        # "変更リードタイム": "{}" # 文字列としてそのままなど
                    })
                )

                st.subheader("価格変動サマリー")
                summary_cols = st.columns(2)
                price_change_df_summary = price_change_df.copy()
                price_change_df_summary["価格変動幅"] = price_change_df_summary["変更後価格"] - price_change_df_summary["変更前価格"]
                summary_by_class = price_change_df_summary.groupby("車両クラス").agg(
                    価格上昇回数=("価格変動幅", lambda x: (x > 0).sum()),
                    価格下降回数=("価格変動幅", lambda x: (x < 0).sum()),
                    総変動回数=("価格変動幅", "count"),
                    平均価格上昇幅=("価格変動幅", lambda x: x[x > 0].mean()),
                    平均価格下降幅=("価格変動幅", lambda x: x[x < 0].mean())
                ).reset_index()
                summary_by_class = summary_by_class.fillna(0)
                with summary_cols[0]:
                    st.write("**クラス別 価格変動回数:**")
                    st.dataframe(summary_by_class[["車両クラス", "価格上昇回数", "価格下降回数", "総変動回数"]].set_index("車両クラス"))
                with summary_cols[1]:
                    st.write("**クラス別 平均価格変動幅:**")
                    st.dataframe(summary_by_class[["車両クラス", "平均価格上昇幅", "平均価格下降幅"]].set_index("車両クラス").style.format("{:.0f}円"))
                overall_rise_count = summary_by_class["価格上昇回数"].sum()
                overall_fall_count = summary_by_class["価格下降回数"].sum()
                st.metric("全クラス合計 価格上昇回数", f"{overall_rise_count} 回")
                st.metric("全クラス合計 価格下降回数", f"{overall_fall_count} 回")
                st.markdown("---")
                
                st.subheader("価格変動 可視化")
                fig_lt_dist_date = plot_price_change_lead_time_distribution(
                    price_change_df, 
                    group_by_col="利用日",
                    title_prefix="利用日別 "
                )
                if fig_lt_dist_date.data:
                    st.plotly_chart(fig_lt_dist_date, use_container_width=True)
                fig_lt_dist_class = plot_price_change_lead_time_distribution(
                    price_change_df, 
                    group_by_col="車両クラス",
                    title_prefix="車両クラス別 "
                )
                if fig_lt_dist_class.data:
                    st.plotly_chart(fig_lt_dist_class, use_container_width=True)
                fig_magnitude_scatter = plot_price_change_magnitude_scatter(
                    price_change_df,
                    car_class_col="車両クラス"
                )
                if fig_magnitude_scatter.data:
                    st.plotly_chart(fig_magnitude_scatter, use_container_width=True)
                
                csv_price_changes = price_change_df.to_csv(index=False).encode('utf-8-sig')
                download_filename_price_changes = f"price_change_details_{price_analysis_start_date}_to_{price_analysis_end_date}.csv"
                st.download_button(
                    label="💾 価格変動点データをダウンロード",
                    data=csv_price_changes,
                    file_name=download_filename_price_changes,
                    mime="text/csv",
                    key="download_price_changes_page_button"
                )
            # else のメッセージは analyze_price_change_details_in_range 内で表示
    else:
        st.info("価格変動分析のためには、データに有効な利用日情報が必要です。") 