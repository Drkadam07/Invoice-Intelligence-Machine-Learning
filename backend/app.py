# run command python -m streamlit run app.py
import streamlit as st
import pandas as pd
from datetime import datetime

from inference.predict_invoice_flag import predict_invoice_flag
from inference.predict_freight import predict_freight_cost


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Invoice Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💰 Invoice Intelligence")

    st.caption("AI-Powered Invoice Analytics")

    st.divider()

    st.subheader("🎯 Prediction Model")

    selected_model = st.radio(
        "Select a model",
        [
            "Invoice Flagging Model",
            "Freight Cost Model",
        ],
        index=0,
    )

    st.divider()

    st.subheader("📌 Application")

    st.info(
        """
        This application uses machine learning to analyze
        vendor invoices and support financial decision-making.
        """
    )

    st.markdown("### Capabilities")

    st.markdown(
        """
        - 🔎 Invoice risk detection
        - 🚚 Freight cost prediction
        - ⚡ Faster invoice processing
        - 💳 Payment decision support
        - 📊 Data-driven analysis
        """
    )

    st.divider()

    st.subheader("🤖 Models")

    if selected_model == "Invoice Flagging Model":

        st.success(
            "Invoice Flagging Model selected"
        )

        st.caption(
            "Identifies invoices that may require additional review."
        )

    else:

        st.success(
            "Freight Cost Model selected"
        )

        st.caption(
            "Estimates freight cost from quantity and invoice value."
        )

    st.divider()

    st.caption(
        f"Invoice Intelligence v1.0 • {datetime.now().year}"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("💰 Invoice Intelligence")

st.subheader(
    "AI-powered vendor invoice analysis and prediction"
)

st.write(
    """
    Use machine learning to identify potentially problematic
    invoices and estimate expected freight costs.
    """
)

st.divider()


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.subheader("📊 Platform Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="🔎 Invoice Analysis",
        value="AI Powered",
    )

with col2:

    st.metric(
        label="🚚 Freight Prediction",
        value="ML Model",
    )

with col3:

    st.metric(
        label="⚡ Processing",
        value="Real Time",
    )

with col4:

    st.metric(
        label="📊 Decision Support",
        value="Data Driven",
    )


st.divider()


# ============================================================
# FREIGHT COST MODEL
# ============================================================

if selected_model == "Freight Cost Model":

    st.header("🚚 Freight Cost Prediction")

    st.write(
        """
        Estimate the expected freight cost based on the
        invoice quantity and total invoice value.
        """
    )

    st.info(
        """
        💡 Enter the invoice quantity and dollar value.
        The trained machine learning model will estimate
        the expected freight cost.
        """
    )

    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader("📋 Invoice Information")

    col1, col2 = st.columns(2)

    with col1:

        quantity = st.number_input(
            "📦 Quantity",
            min_value=0,
            value=10,
            step=1,
            help="Total number of items in the invoice.",
        )

    with col2:

        dollars = st.number_input(
            "💵 Invoice Dollars",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
            help="Total monetary value of the invoice.",
        )

    st.divider()

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("📊 Input Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Quantity",
            f"{quantity:,.0f}",
        )

    with col2:

        st.metric(
            "Invoice Value",
            f"${dollars:,.2f}",
        )

    st.write("")

    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🚚 Predict Freight Cost",
        type="primary",
        use_container_width=True,
    ):

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars],
        }

        try:

            with st.spinner(
                "🤖 Calculating freight cost..."
            ):

                predictions = predict_freight_cost(
                    input_data
                )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if predictions is None:

                st.error(
                    "The model did not return a prediction."
                )

            elif len(predictions) == 0:

                st.error(
                    "The model returned an empty result."
                )

            elif (
                "predicted_freight_cost"
                not in predictions.columns
            ):

                st.error(
                    "The prediction column was not found."
                )

            else:

                predicted_cost = float(
                    predictions[
                        "predicted_freight_cost"
                    ].iloc[0]
                )

                # ============================================
                # SUCCESS
                # ============================================

                st.success(
                    "✅ Freight cost prediction completed successfully."
                )

                st.divider()

                st.subheader(
                    "🎯 Prediction Result"
                )

                # ============================================
                # MAIN RESULT
                # ============================================

                st.metric(
                    "🚚 Predicted Freight Cost",
                    f"${predicted_cost:,.2f}",
                )

                # ============================================
                # RESULT DETAILS
                # ============================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Quantity",
                        f"{quantity:,.0f}",
                    )

                with col2:

                    st.metric(
                        "Invoice Value",
                        f"${dollars:,.2f}",
                    )

                with col3:

                    if dollars > 0:

                        freight_percentage = (
                            predicted_cost
                            / dollars
                        ) * 100

                    else:

                        freight_percentage = 0

                    st.metric(
                        "Freight %",
                        f"{freight_percentage:.2f}%",
                    )

                # ============================================
                # BUSINESS INTERPRETATION
                # ============================================

                st.subheader(
                    "💡 Business Interpretation"
                )

                if freight_percentage <= 5:

                    st.success(
                        "Freight represents a relatively small "
                        "portion of the invoice value."
                    )

                elif freight_percentage <= 10:

                    st.warning(
                        "Freight represents a moderate portion "
                        "of the invoice value."
                    )

                else:

                    st.warning(
                        "Freight represents a relatively high "
                        "portion of the invoice value."
                    )

                # ============================================
                # DETAILS
                # ============================================

                with st.expander(
                    "📊 View Prediction Details"
                ):

                    display_columns = [
                        "Quantity",
                        "Dollars",
                        "predicted_freight_cost",
                    ]

                    available_columns = [
                        column
                        for column in display_columns
                        if column in predictions.columns
                    ]

                    st.dataframe(
                        predictions[
                            available_columns
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as e:

            st.error(
                "❌ Unable to generate freight prediction."
            )

            with st.expander(
                "🔧 Technical Details"
            ):

                st.exception(e)


# ============================================================
# INVOICE FLAGGING MODEL
# ============================================================

else:

    st.header("🔎 Invoice Risk Detection")

    st.write(
        """
        Analyze invoice and receiving information to determine
        whether an invoice may require additional review.
        """
    )

    st.info(
        """
        💡 Provide the invoice and receiving details below.
        The machine learning model will evaluate the information
        and generate an invoice flag prediction.
        """
    )

    # ========================================================
    # TABS
    # ========================================================

    invoice_tab, receiving_tab = st.tabs(
        [
            "📄 Invoice Details",
            "📦 Receiving Details",
        ]
    )

    # ========================================================
    # INVOICE DETAILS
    # ========================================================

    with invoice_tab:

        st.subheader("📄 Invoice Details")

        st.caption(
            "Provide the quantity, monetary value, and freight "
            "information from the invoice."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=0,
                value=10,
                step=1,
                help="Quantity recorded on the invoice.",
            )

        with col2:

            invoice_dollars = st.number_input(
                "💵 Invoice Dollars",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f",
                help="Total invoice amount.",
            )

        with col3:

            freight = st.number_input(
                "🚚 Freight",
                min_value=0.0,
                value=50.0,
                step=10.0,
                format="%.2f",
                help="Freight amount charged on the invoice.",
            )

    # ========================================================
    # RECEIVING DETAILS
    # ========================================================

    with receiving_tab:

        st.subheader("📦 Receiving Details")

        st.caption(
            "Provide information about received quantities, "
            "item values, and average receiving time."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            total_quantity = st.number_input(
                "📦 Total Quantity",
                min_value=0,
                value=15,
                step=1,
                help="Total quantity received.",
            )

        with col2:

            total_item_dollars = st.number_input(
                "💵 Total Item Dollars",
                min_value=0.0,
                value=1200.0,
                step=100.0,
                format="%.2f",
                help="Total value of received items.",
            )

        with col3:

            avg_days_to_receive = st.number_input(
                "📅 Average Days to Receive",
                min_value=0,
                value=5,
                step=1,
                help="Average number of days required to receive items.",
            )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader("📊 Invoice Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Invoice Quantity",
            f"{invoice_quantity:,.0f}",
        )

    with col2:

        st.metric(
            "Invoice Value",
            f"${invoice_dollars:,.2f}",
        )

    with col3:

        st.metric(
            "Freight",
            f"${freight:,.2f}",
        )

    with col4:

        st.metric(
            "Receiving Time",
            f"{avg_days_to_receive} days",
        )

    st.write("")

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Invoice",
        type="primary",
        use_container_width=True,
    ):

        input_data = {
            "invoice_quantity": [
                invoice_quantity
            ],
            "invoice_dollars": [
                invoice_dollars
            ],
            "Freight": [
                freight
            ],
            "total_Quantity": [
                total_quantity
            ],
            "total_item_dollars": [
                total_item_dollars
            ],
            "avg_days_to_receive": [
                avg_days_to_receive
            ],
        }

        try:

            with st.spinner(
                "🤖 Analyzing invoice..."
            ):

                predictions = predict_invoice_flag(
                    input_data
                )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if predictions is None:

                st.error(
                    "The model did not return a prediction."
                )

            elif len(predictions) == 0:

                st.error(
                    "The model returned an empty result."
                )

            elif (
                "predicted_invoice_flag"
                not in predictions.columns
            ):

                st.error(
                    "The prediction column was not found."
                )

            else:

                predicted_flag = predictions[
                    "predicted_invoice_flag"
                ].iloc[0]

                flag_text = str(
                    predicted_flag
                ).strip().upper()

                # ============================================
                # FLAG VALUES
                # ============================================

                flagged_values = {
                    "1",
                    "TRUE",
                    "YES",
                    "Y",
                    "FLAGGED",
                    "RISK",
                    "HIGH",
                }

                is_flagged = (
                    flag_text
                    in flagged_values
                )

                st.divider()

                # ============================================
                # PREDICTION RESULT
                # ============================================

                st.subheader(
                    "🎯 Prediction Result"
                )

                if is_flagged:

                    st.error(
                        """
                        ⚠️ Invoice Requires Review

                        The machine learning model has identified
                        this invoice as potentially requiring
                        additional investigation.
                        """
                    )

                    st.warning(
                        f"Prediction: {predicted_flag}"
                    )

                else:

                    st.success(
                        """
                        ✅ Invoice Appears Normal

                        The machine learning model did not identify
                        this invoice as requiring additional review.
                        """
                    )

                    st.info(
                        f"Prediction: {predicted_flag}"
                    )

                # ============================================
                # KEY METRICS
                # ============================================

                st.subheader(
                    "📊 Invoice Analysis"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Invoice Quantity",
                        f"{invoice_quantity:,.0f}",
                    )

                with col2:

                    st.metric(
                        "Invoice Value",
                        f"${invoice_dollars:,.2f}",
                    )

                with col3:

                    st.metric(
                        "Freight",
                        f"${freight:,.2f}",
                    )

                with col4:

                    st.metric(
                        "Receiving Time",
                        f"{avg_days_to_receive} days",
                    )

                # ============================================
                # BUSINESS INDICATORS
                # ============================================

                st.subheader(
                    "📈 Business Indicators"
                )

                quantity_difference = (
                    total_quantity
                    - invoice_quantity
                )

                dollar_difference = (
                    total_item_dollars
                    - invoice_dollars
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Quantity Difference",
                        f"{quantity_difference:,.0f}",
                    )

                with col2:

                    st.metric(
                        "Value Difference",
                        f"${dollar_difference:,.2f}",
                    )

                with col3:

                    if avg_days_to_receive <= 5:

                        receiving_status = "Normal"

                    elif avg_days_to_receive <= 10:

                        receiving_status = "Moderate"

                    else:

                        receiving_status = "Delayed"

                    st.metric(
                        "Receiving Status",
                        receiving_status,
                    )

                # ============================================
                # RECEIVING ANALYSIS
                # ============================================

                st.subheader(
                    "📦 Receiving Analysis"
                )

                if avg_days_to_receive <= 5:

                    st.success(
                        "Receiving time is within the normal range."
                    )

                elif avg_days_to_receive <= 10:

                    st.warning(
                        "Receiving time is moderately high."
                    )

                else:

                    st.error(
                        "Receiving time is significantly delayed."
                    )

                # ============================================
                # DETAILED DATA
                # ============================================

                with st.expander(
                    "📊 View Detailed Prediction Data"
                ):

                    display_columns = [
                        "invoice_quantity",
                        "invoice_dollars",
                        "Freight",
                        "total_Quantity",
                        "total_item_dollars",
                        "avg_days_to_receive",
                        "predicted_invoice_flag",
                    ]

                    available_columns = [
                        column
                        for column in display_columns
                        if column in predictions.columns
                    ]

                    st.dataframe(
                        predictions[
                            available_columns
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

        except Exception as e:

            st.error(
                "❌ Unable to generate invoice prediction."
            )

            with st.expander(
                "🔧 Technical Details"
            ):

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"""
    © {datetime.now().year} Invoice Intelligence
    • AI-Powered Vendor Invoice Analytics
    • Machine Learning Platform
    """
)
