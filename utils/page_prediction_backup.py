# utils/page_prediction.py

import streamlit as st
import pandas as pd
from typing import Dict, Any
import datetime
from .constants import ( # constants ã‹ã‚‰ã‚¤ãƒ³ãƒãƒ¼ãƒˆ
    TARGET_VARIABLE, DATE_COLUMN, PRICE_COLUMNS, LEAD_TIME_COLUMN,
    CAR_CLASS_COLUMN, BOOKING_DATE_COLUMN, USAGE_COUNT_COLUMN,
    LAG_TARGET_COLUMN, LAG_DAYS
)
from .data_processing import ( # ç›¸å¯¾ã‚¤ãƒ³ãƒãƒ¼ãƒˆã«å¤‰æ›´
    filter_data_by_date, create_scenario_data,
    find_last_price_change_lead_time
)
from .visualization import ( # ç›¸å¯¾ã‚¤ãƒ³ãƒãƒ¼ãƒˆã«å¤‰æ›´
    plot_booking_curve, plot_price_trends, plot_comparison_curve,
    plot_feature_importance
)
from .modeling import ( # ç›¸å¯¾ã‚¤ãƒ³ãƒãƒ¼ãƒˆã«å¤‰æ›´
    setup_and_compare_models, predict_with_model, get_feature_importance_df
)
from .ui_components import ( # ç›¸å¯¾ã‚¤ãƒ³ãƒãƒ¼ãƒˆã«å¤‰æ›´
    render_prediction_sidebar_widgets
)
from .model_storage import load_model, list_saved_models, load_comparison_results, get_model_metadata, prepare_features_for_prediction
import numpy as np

# --- äºˆæ¸¬ãƒ»æ¯”è¼ƒåˆ†æžãƒšãƒ¼ã‚¸æç”»é–¢æ•° ---
def render_prediction_analysis_page(data: pd.DataFrame, config: Dict[str, Any]):
    st.title("äºˆæ¸¬åˆ†æž")

    # --- ã‚µã‚¤ãƒ‰ãƒãƒ¼ã‚¦ã‚£ã‚¸ã‚§ãƒƒãƒˆã®æç”»ã¨å€¤ã®å–å¾— ---
    (
        selected_car_class,
        selected_model_info,
        _  # äºˆæ¸¬å®Ÿè¡Œãƒœã‚¿ãƒ³ã¯ãƒ¡ã‚¤ãƒ³ã‚¨ãƒªã‚¢ã§è¡¨ç¤ºã™ã‚‹ãŸã‚ç„¡è¦–
    ) = render_prediction_sidebar_widgets(data)

    # --- ãƒ¡ã‚¤ãƒ³ã‚¨ãƒªã‚¢ --- #
    if not selected_model_info:
        st.warning("äºˆæ¸¬ã‚’å®Ÿè¡Œã™ã‚‹ã«ã¯ã€ã¾ãšã€Œãƒ¢ãƒ‡ãƒ«ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°ã€ãƒšãƒ¼ã‚¸ã§ãƒ¢ãƒ‡ãƒ«ã‚’ä½œæˆã—ã€ã‚µã‚¤ãƒ‰ãƒãƒ¼ã§ä½¿ç”¨ã™ã‚‹ãƒ¢ãƒ‡ãƒ«ã‚’é¸æŠžã—ã¦ãã ã•ã„ã€‚")
        return

    # ãƒ¢ãƒ‡ãƒ«æ¯”è¼ƒçµæžœã®è©³ç´°ã‚»ã‚¯ã‚·ãƒ§ãƒ³ï¼ˆç‹¬ç«‹ã—ãŸã‚»ã‚¯ã‚·ãƒ§ãƒ³ã¨ã—ã¦è¡¨ç¤ºï¼‰
    st.header("ãƒ¢ãƒ‡ãƒ«æ¯”è¼ƒçµæžœã®è©³ç´°")
    comparison_path = selected_model_info.get("comparison_results_path")
    if comparison_path:
        comparison_results = load_comparison_results(comparison_path)
        if comparison_results is not None:
            st.dataframe(comparison_results)
        else:
            st.warning("æ¯”è¼ƒçµæžœã‚’èª­ã¿è¾¼ã‚ã¾ã›ã‚“ã§ã—ãŸã€‚")
    else:
        st.info("ã“ã®ãƒ¢ãƒ‡ãƒ«ã«ã¯æ¯”è¼ƒçµæžœãŒä¿å­˜ã•ã‚Œã¦ã„ã¾ã›ã‚“ã€‚")
    
    st.markdown("---") # æ¯”è¼ƒçµæžœã¨æ¬¡ã®ã‚»ã‚¯ã‚·ãƒ§ãƒ³ã®é–“ã«åŒºåˆ‡ã‚Šç·š

    # --- åˆ†æžæ—¥ã®é¸æŠžï¼ˆãƒ¢ãƒ‡ãƒ«æ¯”è¼ƒçµæžœã®å¾Œã«è¡¨ç¤ºï¼‰ ---
    st.header("åˆ©ç”¨æ—¥ã®é¸æŠž")
    
    # é¸æŠžã•ã‚ŒãŸè»Šä¸¡ã‚¯ãƒ©ã‚¹ã§ãƒ‡ãƒ¼ã‚¿ã‚’ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚° (æ—¥ä»˜é¸æŠžç”¨)
    if selected_car_class == "å…¨ã‚¯ãƒ©ã‚¹":
        data_for_date_selection = data
    else:
        data_for_date_selection = data[data[CAR_CLASS_COLUMN] == selected_car_class]

    # åˆ©ç”¨æ—¥ã®é¸æŠžã‚¦ã‚£ã‚¸ã‚§ãƒƒãƒˆ
    selected_date = None
    if DATE_COLUMN in data_for_date_selection.columns and pd.api.types.is_datetime64_any_dtype(data_for_date_selection[DATE_COLUMN]):
        available_dates = data_for_date_selection[DATE_COLUMN].dt.date.unique()
        if len(available_dates) > 0:
            date_options_str = ['æ—¥ä»˜ã‚’é¸æŠž'] + sorted([d.strftime('%Y-%m-%d') for d in available_dates])
            selected_date_str = st.selectbox(
                f"'{DATE_COLUMN}'ã‚’é¸æŠž:",
                options=date_options_str, index=0, key="pred_date_select"
            )
            if selected_date_str != 'æ—¥ä»˜ã‚’é¸æŠž':
                try:
                    selected_date = pd.to_datetime(selected_date_str).date()
                except ValueError:
                    st.error("é¸æŠžã•ã‚ŒãŸæ—¥ä»˜ã®å½¢å¼ãŒç„¡åŠ¹ã§ã™ã€‚")
                    selected_date = None # Noneã«ãƒªã‚»ãƒƒãƒˆ
        else:
            st.info(f"'{selected_car_class}'ã‚¯ãƒ©ã‚¹ã«ã¯æœ‰åŠ¹ãª'{DATE_COLUMN}'ãƒ‡ãƒ¼ã‚¿ãŒã‚ã‚Šã¾ã›ã‚“ã€‚")
    else:
        st.warning(f"'{DATE_COLUMN}'åˆ—ãŒãªã„ã‹æ—¥ä»˜åž‹ã§ã¯ã‚ã‚Šã¾ã›ã‚“ã€‚")
        selected_date = None # Noneã«ãƒªã‚»ãƒƒãƒˆ

    # äºˆæ¸¬å®Ÿè¡Œãƒœã‚¿ãƒ³ï¼ˆãƒ¢ãƒ‡ãƒ«ã¨åˆ©ç”¨æ—¥ã®ä¸¡æ–¹ãŒé¸æŠžã•ã‚ŒãŸå ´åˆã«è¡¨ç¤ºï¼‰
    run_prediction = False
    if selected_model_info and selected_date:
        run_prediction = st.button("ðŸ”® äºˆæ¸¬å®Ÿè¡Œ", key="run_prediction")
    
    st.markdown("---") # æ—¥ä»˜é¸æŠžã¨æ¬¡ã®ã‚»ã‚¯ã‚·ãƒ§ãƒ³ã®é–“ã«åŒºåˆ‡ã‚Šç·š

    # å‡¦ç†æ¸ˆã¿ãƒ‡ãƒ¼ã‚¿ãƒ—ãƒ¬ãƒ“ãƒ¥ãƒ¼ï¼ˆæ—¥ä»˜é¸æŠžã®å¾Œã«è¡¨ç¤ºï¼‰
    st.subheader("å‡¦ç†æ¸ˆã¿ãƒ‡ãƒ¼ã‚¿ãƒ—ãƒ¬ãƒ“ãƒ¥ãƒ¼ (ç¾åœ¨ã®çŠ¶æ…‹)")
    st.dataframe(data.head())

    st.markdown("---") # ãƒ‡ãƒ¼ã‚¿ãƒ—ãƒ¬ãƒ“ãƒ¥ãƒ¼ã¨æ¬¡ã®ã‚»ã‚¯ã‚·ãƒ§ãƒ³ã®é–“ã«åŒºåˆ‡ã‚Šç·š

    if selected_date is not None:
        st.header(f"åˆ†æžçµæžœ: {selected_date} ({selected_car_class})")

        # ãƒ‡ãƒ¼ã‚¿ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚°
        if selected_car_class == "å…¨ã‚¯ãƒ©ã‚¹":
            data_class_filtered_for_viz = data
        else:
            data_class_filtered_for_viz = data[data[CAR_CLASS_COLUMN] == selected_car_class]

        data_filtered = filter_data_by_date(data_class_filtered_for_viz, DATE_COLUMN, selected_date)

        if not data_filtered.empty:
            if LEAD_TIME_COLUMN in data_filtered.columns:
                data_filtered_sorted = data_filtered.sort_values(by=LEAD_TIME_COLUMN)

                # å®Ÿéš›ã®äºˆç´„æ›²ç·šã¨ä¾¡æ ¼æŽ¨ç§»ã‚°ãƒ©ãƒ•
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("å®Ÿéš›ã®äºˆç´„æ›²ç·š")
                    fig_actual = plot_booking_curve(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_col=TARGET_VARIABLE, title=f"{selected_date} {selected_car_class} å®Ÿéš›ã®äºˆç´„æ›²ç·š")
                    st.plotly_chart(fig_actual, use_container_width=True)
                with col2:
                    st.subheader("ä¾¡æ ¼æŽ¨ç§»")
                    fig_prices = plot_price_trends(data_filtered_sorted, x_col=LEAD_TIME_COLUMN, y_cols=PRICE_COLUMNS, title=f"{selected_date} {selected_car_class} ä¾¡æ ¼æŽ¨ç§»")
                    st.plotly_chart(fig_prices, use_container_width=True)

                # äºˆæ¸¬å®Ÿè¡Œã‚»ã‚¯ã‚·ãƒ§ãƒ³
                if run_prediction:
                    st.markdown("---")
                    st.header("äºˆæ¸¬å®Ÿè¡Œ")
                    with st.spinner('äºˆæ¸¬ã‚’å®Ÿè¡Œä¸­...'):
                        # ãƒ¢ãƒ‡ãƒ«ã®ãƒ­ãƒ¼ãƒ‰
                        model = load_model(selected_model_info["path"])
                        
                        if model is None:
                            st.error("ãƒ¢ãƒ‡ãƒ«ã®èª­ã¿è¾¼ã¿ã«å¤±æ•—ã—ã¾ã—ãŸã€‚")
                            return
                        
                        # ã‚·ãƒŠãƒªã‚ªäºˆæ¸¬
                        last_change_lt = find_last_price_change_lead_time(data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN)
                        if last_change_lt is not None:
                            st.write(f"ä¾¡æ ¼æœ€çµ‚å¤‰æ›´ãƒªãƒ¼ãƒ‰ã‚¿ã‚¤ãƒ : {last_change_lt}")
                            data_scenario = create_scenario_data(
                                data_filtered_sorted, PRICE_COLUMNS, LEAD_TIME_COLUMN,
                                scenario_type='last_change_fixed', change_lead_time=last_change_lt
                            )
                            scenario_title_suffix = f"ä¾¡æ ¼æœ€çµ‚å¤‰æ›´ç‚¹(LT={last_change_lt})å›ºå®šã‚·ãƒŠãƒªã‚ª"

                            if not data_scenario.empty:
                                with st.expander(f"äºˆæ¸¬ã«ä½¿ç”¨ã—ãŸãƒ‡ãƒ¼ã‚¿ ({scenario_title_suffix}) ã®ã‚µãƒ³ãƒ—ãƒ«"):
                                    # selected_featuresã‚’ãƒ¢ãƒ‡ãƒ«ã®ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã‹ã‚‰å–å¾—
                                    selected_features = selected_model_info.get("selected_features", [])
                                    display_columns = selected_features.copy() if selected_features else []
                                    
                                    # å¿…é ˆåˆ—ã‚’è¿½åŠ  (é‡è¤‡ã—ãªã„ã‚ˆã†ã«ãƒã‚§ãƒƒã‚¯)
                                    essential_columns = [LEAD_TIME_COLUMN, TARGET_VARIABLE] + PRICE_COLUMNS
                                    for col in essential_columns:
                                        if col in data_scenario.columns and col not in display_columns:
                                            display_columns.append(col)
                                    
                                    # ãƒ©ã‚°åˆ—åã‚’è¿½åŠ  (é‡è¤‡ã—ãªã„ã‚ˆã†ã«ãƒã‚§ãƒƒã‚¯)
                                    lag_col_name = f"{LAG_TARGET_COLUMN}_lag{LAG_DAYS}"
                                    if lag_col_name in data_scenario.columns and lag_col_name not in display_columns:
                                         display_columns.append(lag_col_name)

                                    # å­˜åœ¨ã™ã‚‹åˆ—ã®ã¿ã«ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚° & æœ€çµ‚é‡è¤‡å‰Šé™¤
                                    existing_display_columns = [col for col in display_columns if col in data_scenario.columns]
                                    if pd.Series(existing_display_columns).duplicated().any():
                                        existing_display_columns = pd.Series(existing_display_columns).drop_duplicates().tolist()

                                    if existing_display_columns:
                                        try:
                                            st.dataframe(data_scenario[existing_display_columns].head())
                                            csv = data_scenario[existing_display_columns].to_csv(index=False).encode('utf-8')
                                            filename = f"scenario_data_{selected_date}_{selected_car_class}_{scenario_title_suffix.replace(' ', '_')}.csv"
                                            st.download_button("ðŸ’¾ äºˆæ¸¬ç”¨ãƒ‡ãƒ¼ã‚¿ã‚’ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰", csv, filename, "text/csv")
                                        except ValueError as e:
                                             st.error(f"ã‚·ãƒŠãƒªã‚ªãƒ‡ãƒ¼ã‚¿ã‚µãƒ³ãƒ—ãƒ«è¡¨ç¤ºä¸­ã«ã‚¨ãƒ©ãƒ¼: {e}")
                                             st.write("è¡¨ç¤ºã—ã‚ˆã†ã¨ã—ãŸåˆ—:", existing_display_columns)
                                    else:
                                        st.warning("è¡¨ç¤ºåˆ—ãªã—")

                                # ãƒ¢ãƒ‡ãƒ«ã®ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã‚’å–å¾—
                                model_filename = selected_model_info.get("filename")
                                model_metadata = None
                                
                                # ãƒ‡ãƒãƒƒã‚°ãƒ­ã‚°ã‚’ã‚¨ã‚¯ã‚¹ãƒ‘ãƒ³ãƒ€ãƒ¼ã«ç§»å‹•
                                with st.expander("äºˆæ¸¬ãƒ—ãƒ­ã‚»ã‚¹ã®ãƒ‡ãƒãƒƒã‚°æƒ…å ±", expanded=False):
                                    st.info("ãƒ¢ãƒ‡ãƒ«ã®å†…éƒ¨æ§‹é€ ã‹ã‚‰ç‰¹å¾´é‡æƒ…å ±ã‚’æ¤œæŸ»ä¸­...")
                                    
                                    # *** æ–°ã—ã„æ–¹æ³•: ãƒ¢ãƒ‡ãƒ«è‡ªä½“ã‹ã‚‰ç‰¹å¾´é‡åã‚’ç›´æŽ¥å–å¾—ã™ã‚‹ ***
                                    try:
                                        # ã‚¹ã‚­ãƒƒãƒ—ã•ã‚Œã‚‹å¯èƒ½æ€§ãŒã‚ã‚‹ä¸€èˆ¬çš„ãªãƒ¢ãƒ‡ãƒ«ãƒ»ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³å±žæ€§
                                        skipped_attrs = ['_isfit', 'classes_', 'n_classes_', 'n_features_in_', 'base_score', 'label_encoder', 'estimator']
                                        
                                        # ãƒ¢ãƒ‡ãƒ«å±žæ€§ã®æ¤œæŸ»
                                        model_attrs = dir(model)
                                        feature_attrs = [attr for attr in model_attrs 
                                                       if 'feature' in attr.lower() and attr not in skipped_attrs]
                                        
                                        # ç‰¹å¾´é‡åã‚’æ¤œå‡º
                                        model_features = None
                                        if hasattr(model, 'feature_names_in_'):
                                            model_features = list(model.feature_names_in_)
                                            st.success(f"ãƒ¢ãƒ‡ãƒ«ã‹ã‚‰ç›´æŽ¥ {len(model_features)}å€‹ã®ç‰¹å¾´é‡åã‚’å–å¾—ã—ã¾ã—ãŸ (feature_names_in_)")
                                        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                                            # XGBoostã®å ´åˆ
                                            model_features = model.get_booster().feature_names
                                            st.success(f"XGBoostãƒ¢ãƒ‡ãƒ«ã‹ã‚‰ {len(model_features)}å€‹ã®ç‰¹å¾´é‡åã‚’å–å¾—ã—ã¾ã—ãŸ")
                                        elif hasattr(model, 'steps'):
                                            # ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ã®å ´åˆã¯æœ€å¾Œã®ã‚¹ãƒ†ãƒƒãƒ—ã‚’ç¢ºèª
                                            final_step = model.steps[-1][1]
                                            if hasattr(final_step, 'feature_names_in_'):
                                                model_features = list(final_step.feature_names_in_)
                                                st.success(f"ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ã®æœ€çµ‚ã‚¹ãƒ†ãƒƒãƒ—ã‹ã‚‰ {len(model_features)}å€‹ã®ç‰¹å¾´é‡åã‚’å–å¾—ã—ã¾ã—ãŸ")
                                        
                                        # ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã®ç‰¹å¾´é‡åã‚’ä¸Šæ›¸ã
                                        if model_features:
                                            if model_filename:
                                                model_metadata = get_model_metadata(model_filename)
                                                if model_metadata:
                                                    # å…ƒã®model_columnsã‚’ä¿å­˜
                                                    original_columns = model_metadata.get("model_columns", [])
                                                    if original_columns:
                                                        st.info(f"ãƒ¡ã‚¿ãƒ‡ãƒ¼ã‚¿ã®ç‰¹å¾´é‡å ({len(original_columns)}å€‹) ã‚’ãƒ¢ãƒ‡ãƒ«ã‹ã‚‰å–å¾—ã—ãŸç‰¹å¾´é‡åã§ä¸Šæ›¸ãã—ã¾ã™")
                                                    
                                                    # model_featuresã§ä¸Šæ›¸ã
                                                    model_metadata["model_columns"] = model_features
                                        
                                        # ãƒ¢ãƒ‡ãƒ«æ§‹é€ ã®ãã®ä»–ã®æƒ…å ±ã‚’è¡¨ç¤º
                                        st.info(f"ãƒ¢ãƒ‡ãƒ«ã®åž‹: {type(model).__name__}")
                                        if feature_attrs:
                                            st.info(f"æ¤œå‡ºã•ã‚ŒãŸç‰¹å¾´é‡é–¢é€£ã®å±žæ€§: {feature_attrs}")
                                    
                                    except Exception as e:
                                        st.warning(f"ãƒ¢ãƒ‡ãƒ«ã®å†…éƒ¨æ§‹é€ æ¤œæŸ»ä¸­ã«ã‚¨ãƒ©ãƒ¼: {e}")
                                    
                                    if not model_metadata and model_filename:
                                        model_metadata = get_model_metadata(model_filename)
                                    
                                    if model_metadata:
                                        if "model_columns" in model_metadata:
                                            st.success(f"ç‰¹å¾´é‡æƒ…å ±ãŒå­˜åœ¨ã—ã¾ã™: {len(model_metadata['model_columns'])}å€‹ã®åˆ—")
                                            
                                            # ç‰¹å¾´é‡æƒ…å ±ã®ä¾‹ã‚’è¡¨ç¤º
                                            st.subheader("ç‰¹å¾´é‡æƒ…å ±ã‚µãƒ³ãƒ—ãƒ«")
                                            st.json(model_metadata["model_columns"][:10])
                                        else:
                                            st.warning("ç‰¹å¾´é‡æƒ…å ±ãŒè¦‹ã¤ã‹ã‚Šã¾ã›ã‚“ã€‚äºˆæ¸¬ãŒå¤±æ•—ã™ã‚‹å¯èƒ½æ€§ãŒã‚ã‚Šã¾ã™ã€‚")
                                
                                # äºˆæ¸¬æ™‚ã®ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯æ–¹æ³•: ãƒ€ã‚¤ãƒ¬ã‚¯ãƒˆäºˆæ¸¬ã‚’è©¦ã¿ã‚‹
                                try:
                                    # äºˆæ¸¬ãƒ—ãƒ­ã‚»ã‚¹ã®é–‹å§‹ã‚’é€šçŸ¥
                                    st.info("äºˆæ¸¬å‡¦ç†ã‚’é–‹å§‹ã—ã¾ã™...")
                                    
                                    # ãƒ‡ãƒãƒƒã‚°ãƒ­ã‚°ã‚’ã‚¨ã‚¯ã‚¹ãƒ‘ãƒ³ãƒ€ãƒ¼ã«ç§»å‹•
                                    with st.expander("äºˆæ¸¬å®Ÿè¡Œè©³ç´°ãƒ­ã‚°", expanded=False):
                                        st.info("ãƒ¢ãƒ‡ãƒ«ã®predict()ãƒ¡ã‚½ãƒƒãƒ‰ã‚’ç›´æŽ¥ä½¿ç”¨ã—ã¦äºˆæ¸¬ã‚’å®Ÿè¡Œã—ã¾ã™...")
                                        
                                        # æ—¥ä»˜åˆ—ã‚’æ–‡å­—åˆ—ã«å¤‰æ›ï¼ˆæœ€çµ‚æ‰‹æ®µï¼‰
                                        scen_data_transformed = data_scenario.copy()
                                        date_cols = scen_data_transformed.select_dtypes(include=['datetime64']).columns
                                        for col in date_cols:
                                            scen_data_transformed[col] = scen_data_transformed[col].dt.strftime('%Y-%m-%d')
                                        
                                        # ã‚¿ãƒ¼ã‚²ãƒƒãƒˆå¤‰æ•°ãŒã‚ã‚‹å ´åˆã¯é™¤åŽ»
                                        if TARGET_VARIABLE in scen_data_transformed.columns:
                                            X = scen_data_transformed.drop(columns=[TARGET_VARIABLE])
                                        else:
                                            X = scen_data_transformed
                                        
                                        # ç›´æŽ¥äºˆæ¸¬
                                        y_pred = None
                                        
                                        # 1. é€šå¸¸ã®predictã‚’è©¦è¡Œ
                                        try:
                                            st.info("ãƒ¢ãƒ‡ãƒ«ã®ç›´æŽ¥äºˆæ¸¬ã‚’è©¦è¡Œ...")
                                            if hasattr(model, 'predict'):
                                                y_pred = model.predict(X)
                                                st.success("ç›´æŽ¥äºˆæ¸¬æˆåŠŸ!")
                                        except Exception as e1:
                                            st.error(f"ç›´æŽ¥äºˆæ¸¬ã§ã‚¨ãƒ©ãƒ¼: {e1}")
                                            
                                            # 2. ç‰¹å¾´é‡å¤‰æ›ã‚’å®Ÿè¡Œã—ã¦å†åº¦è©¦è¡Œ
                                            try:
                                                if model_metadata and "model_columns" in model_metadata:
                                                    st.info("ç‰¹å¾´é‡å¤‰æ›ã‚’é©ç”¨ã—ã¦å†è©¦è¡Œ...")
                                                    transformed_data = prepare_features_for_prediction(X, model_metadata)
                                                    y_pred = model.predict(transformed_data)
                                                    st.success("ç‰¹å¾´é‡å¤‰æ›å¾Œã®äºˆæ¸¬æˆåŠŸ!")
                                            except Exception as e2:
                                                st.error(f"ç‰¹å¾´é‡å¤‰æ›å¾Œã®äºˆæ¸¬ã§ã‚‚ã‚¨ãƒ©ãƒ¼: {e2}")
                                    
                                    # äºˆæ¸¬çµæžœã‚’åˆ©ç”¨
                                    if y_pred is not None:
                                        st.success("äºˆæ¸¬å®Œäº†!")
                                        
                                        # çµæžœã‚’ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã«å¤‰æ›
                                        predictions_result = data_scenario.copy()
                                        predictions_result['prediction_label'] = y_pred
                                        
                                        # æ¯”è¼ƒã‚°ãƒ©ãƒ•ã¨è¡¨ã‚’è¡¨ç¤º
                                        st.markdown("---")
                                        st.subheader(f"å®Ÿç¸¾ vs äºˆæ¸¬æ¯”è¼ƒ ({scenario_title_suffix}) - LT {last_change_lt} ä»¥é™")
                                        
                                        actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                        predictions_filtered_display = predictions_result[predictions_result[LEAD_TIME_COLUMN] <= last_change_lt]
                                        
                                        if not actual_filtered_display.empty and not predictions_filtered_display.empty:
                                            fig_compare = plot_comparison_curve(
                                                df_actual=actual_filtered_display, 
                                                df_predicted=predictions_filtered_display,
                                                x_col=LEAD_TIME_COLUMN, 
                                                y_actual_col=TARGET_VARIABLE, 
                                                y_pred_col='prediction_label',
                                                title=f"{selected_date} {selected_car_class} å®Ÿç¸¾ vs äºˆæ¸¬ (LT {last_change_lt} ä»¥é™)"
                                            )
                                            st.plotly_chart(fig_compare, use_container_width=True)
                                            
                                            st.subheader(f"å®Ÿç¸¾ vs äºˆæ¸¬ ãƒ‡ãƒ¼ã‚¿ãƒ†ãƒ¼ãƒ–ãƒ« (LT {last_change_lt} ä»¥é™)")
                                            df_actual_for_table = actual_filtered_display[[LEAD_TIME_COLUMN, TARGET_VARIABLE]].rename(columns={TARGET_VARIABLE: 'å®Ÿç¸¾åˆ©ç”¨å°æ•°'})
                                            df_pred_for_table = predictions_filtered_display[[LEAD_TIME_COLUMN, 'prediction_label']].rename(columns={'prediction_label': 'äºˆæ¸¬åˆ©ç”¨å°æ•°'})
                                            df_comparison_table = pd.merge(df_actual_for_table, df_pred_for_table, on=LEAD_TIME_COLUMN, how='inner')
                                            st.dataframe(df_comparison_table.sort_values(by=LEAD_TIME_COLUMN).reset_index(drop=True))
                                        else:
                                            st.warning(f"LT {last_change_lt} ä»¥é™ã®è¡¨ç¤ºãƒ‡ãƒ¼ã‚¿ãŒã‚ã‚Šã¾ã›ã‚“")
                                    else:
                                        st.error("ã™ã¹ã¦ã®äºˆæ¸¬æ–¹æ³•ãŒå¤±æ•—ã—ã¾ã—ãŸ")
                                
                                except Exception as e:
                                    st.error(f"äºˆæ¸¬å‡¦ç†ä¸­ã«ã‚¨ãƒ©ãƒ¼: {e}")
                                    
                                    # å¾“æ¥ã®æ–¹æ³•ã‚’ã‚¨ã‚¯ã‚¹ãƒ‘ãƒ³ãƒ€ãƒ¼ã«ç§»å‹•
                                    with st.expander("ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯äºˆæ¸¬æ–¹æ³•ã®è©³ç´°ãƒ­ã‚°", expanded=False):
                                        st.warning("ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯äºˆæ¸¬æ–¹æ³•ã‚’è©¦è¡Œã—ã¾ã™...")
                                        # å¾“æ¥ã®æ–¹æ³•ã‚’è©¦ã¿ã‚‹
                                        if model_metadata and "model_columns" in model_metadata:
                                            # ç‰¹å¾´é‡å¤‰æ›ã‚’é©ç”¨
                                            st.info("äºˆæ¸¬ãƒ‡ãƒ¼ã‚¿ã‚’å­¦ç¿’æ™‚ã®ç‰¹å¾´é‡å½¢å¼ã«å¤‰æ›ã—ã¾ã™...")
                                            data_for_prediction = prepare_features_for_prediction(data_scenario, model_metadata)
                                            # äºˆæ¸¬å®Ÿè¡Œ
                                            predictions, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation = predict_with_model(model, data_for_prediction, target=TARGET_VARIABLE)
                                        else:
                                            # ç‰¹å¾´é‡å¤‰æ›ãªã—ã§äºˆæ¸¬ã‚’è©¦è¡Œ
                                            st.warning("ç‰¹å¾´é‡æƒ…å ±ãŒãªã„ãŸã‚ã€ç‰¹å¾´é‡å¤‰æ›ãªã—ã§äºˆæ¸¬ã‚’è©¦è¡Œã—ã¾ã™ã€‚ã‚¨ãƒ©ãƒ¼ãŒç™ºç”Ÿã™ã‚‹å¯èƒ½æ€§ãŒã‚ã‚Šã¾ã™ã€‚")
                                            predictions, imputation_log, nan_rows_before_imputation, nan_rows_after_imputation = predict_with_model(model, data_scenario, target=TARGET_VARIABLE)

                                        # è£œå®Œãƒ­ã‚°ãŒã‚ã‚Œã°ãƒ†ãƒ¼ãƒ–ãƒ«ã§è¡¨ç¤º
                                    if imputation_log:
                                        st.subheader("äºˆæ¸¬å‰ã®ç‰¹å¾´é‡æ¬ æå€¤è£œå®Œã®è©³ç´°")
                                        log_df = pd.DataFrame(imputation_log)
                                        if 'Imputation Value' in log_df.columns:
                                             log_df['Imputation Value'] = log_df['Imputation Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                                        st.dataframe(log_df)
                                            
                                        if nan_rows_before_imputation is not None and not nan_rows_before_imputation.empty:
                                            st.subheader("NaNå€¤ãŒå«ã¾ã‚Œã¦ã„ãŸè¡Œï¼ˆè£œå®Œå‰ï¼‰")
                                            st.dataframe(nan_rows_before_imputation)
                                            
                                        if nan_rows_after_imputation is not None and not nan_rows_after_imputation.empty:
                                            st.subheader("NaNå€¤ãŒå«ã¾ã‚Œã¦ã„ãŸè¡Œï¼ˆè£œå®Œå¾Œï¼‰")
                                            st.dataframe(nan_rows_after_imputation)

                                    # ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯çµæžœã®è¡¨ç¤º
                                    if 'predictions' in locals() and not predictions.empty:
                                        st.success("ãƒ•ã‚©ãƒ¼ãƒ«ãƒãƒƒã‚¯äºˆæ¸¬ãŒæˆåŠŸã—ã¾ã—ãŸï¼")
                                        # çµæžœè¡¨ç¤º
                                        st.markdown("---")
                                        
                                        # å®Ÿç¸¾ vs äºˆæ¸¬ã®æ¯”è¼ƒã‚°ãƒ©ãƒ•ã¨è¡¨
                                        st.subheader(f"å®Ÿç¸¾ vs äºˆæ¸¬æ¯”è¼ƒ ({scenario_title_suffix}) - LT {last_change_lt} ä»¥é™")
                                        actual_filtered_display = data_filtered_sorted[data_filtered_sorted[LEAD_TIME_COLUMN] <= last_change_lt]
                                        if LEAD_TIME_COLUMN not in predictions.columns:
                                            predictions_with_lt = pd.merge(predictions, data_scenario[[LEAD_TIME_COLUMN]], left_index=True, right_index=True, how='left')
                                        else:
                                            predictions_with_lt = predictions
                                        predictions_filtered_display = predictions_with_lt[predictions_with_lt[LEAD_TIME_COLUMN] <= last_change_lt]

                                        if not actual_filtered_display.empty and not predictions_filtered_display.empty:
                                            fig_compare = plot_comparison_curve(
                                                df_actual=actual_filtered_display, df_predicted=predictions_filtered_display,
                                                x_col=LEAD_TIME_COLUMN, y_actual_col=TARGET_VARIABLE, y_pred_col='prediction_label',
                                                title=f"{selected_date} {selected_car_class} å®Ÿç¸¾ vs äºˆæ¸¬ (LT {last_change_lt} ä»¥é™)"
                                            )
                                            st.plotly_chart(fig_compare, use_container_width=True)

                                            st.subheader(f"å®Ÿç¸¾ vs äºˆæ¸¬ ãƒ‡ãƒ¼ã‚¿ãƒ†ãƒ¼ãƒ–ãƒ« (LT {last_change_lt} ä»¥é™)")
                                            df_actual_for_table = actual_filtered_display[[LEAD_TIME_COLUMN, TARGET_VARIABLE]].rename(columns={TARGET_VARIABLE: 'å®Ÿç¸¾åˆ©ç”¨å°æ•°'})
                                            df_pred_for_table = predictions_filtered_display[[LEAD_TIME_COLUMN, 'prediction_label']].rename(columns={'prediction_label': 'äºˆæ¸¬åˆ©ç”¨å°æ•°'})
                                            df_comparison_table = pd.merge(df_actual_for_table, df_pred_for_table, on=LEAD_TIME_COLUMN, how='inner')
                                            st.dataframe(df_comparison_table.sort_values(by=LEAD_TIME_COLUMN).reset_index(drop=True))
                                        else:
                                            st.warning(f"LT {last_change_lt} ä»¥é™ã®è¡¨ç¤ºãƒ‡ãƒ¼ã‚¿ãªã—")
                                    else:
                                        st.error("ã™ã¹ã¦ã®äºˆæ¸¬æ–¹æ³•ãŒå¤±æ•—ã—ã¾ã—ãŸ")
                                else:
                                    st.error("ã‚·ãƒŠãƒªã‚ªãƒ‡ãƒ¼ã‚¿ä½œæˆå¤±æ•—")
                            else:
                                st.warning("ä¾¡æ ¼å¤‰å‹•ãŒè¦‹ã¤ã‹ã‚‰ãªã‹ã£ãŸãŸã‚ã€æœ€çµ‚ä¾¡æ ¼å›ºå®šã‚·ãƒŠãƒªã‚ªã§ã®äºˆæ¸¬ã¯å®Ÿè¡Œã§ãã¾ã›ã‚“ã€‚")
            else:
                st.warning(f"'{LEAD_TIME_COLUMN}'åˆ—ãªã—")
        else:
             st.info(f"'{selected_date}' ({selected_car_class}) ã®ãƒ‡ãƒ¼ã‚¿ãªã—")
    elif selected_date is None:
        # æ—¥ä»˜é¸æŠžãƒ•ã‚£ãƒ¼ãƒ«ãƒ‰ã®å­˜åœ¨ç¢ºèª
        if DATE_COLUMN in data.columns and pd.api.types.is_datetime64_any_dtype(data[DATE_COLUMN]) and not data[DATE_COLUMN].isnull().all():
             st.info("åˆ†æžçµæžœã‚’è¡¨ç¤ºã™ã‚‹ã«ã¯ã€åˆ©ç”¨æ—¥ã‚’é¸æŠžã—ã¦ãã ã•ã„ã€‚") 
