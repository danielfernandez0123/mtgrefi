import streamlit as st
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(
    page_title="Mortgage Refinancing Calculator",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Advanced Mortgage Refinancing Calculator")
st.markdown("""
This calculator helps you determine the optimal time to refinance your mortgage using advanced financial models including:
- **Option Model**: Calculates optimal refinancing trigger rates
- **ENPV Analysis**: Expected Net Present Value with prepayment probability
- **Break-even Analysis**: Shows when refinancing pays off
""")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Input Parameters")
    
    # Option Model Inputs
    st.subheader("Option Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        i0 = st.number_input("Current Mortgage Rate (%)", 
                            value=6.0, min_value=0.0, max_value=20.0, step=0.1) / 100
        rho = st.number_input("Real Discount Rate (%)", 
                             value=4.0, min_value=0.0, max_value=20.0, step=0.1) / 100
        lam = st.number_input("Annual CPR (%)", 
                             value=16.0, min_value=0.0, max_value=100.0, step=0.5) / 100
        sigma = st.number_input("Rate Volatility", 
                               value=0.007, min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
        tau = st.number_input("Marginal Tax Rate (%)", 
                             value=0.0, min_value=0.0, max_value=50.0, step=1.0) / 100
    
    with col2:
        M = st.number_input("Mortgage Balance ($)", 
                           value=500000, min_value=10000, max_value=10000000, step=10000)
        p = st.number_input("Points (% of balance)", 
                           value=0.0, min_value=0.0, max_value=5.0, step=0.1) / 100
        F = st.number_input("Fixed Fees ($)", 
                           value=10000, min_value=0, max_value=100000, step=500)
        Gamma = st.number_input("Remaining Years", 
                               value=20.0, min_value=1.0, max_value=30.0, step=0.5)
    
    # ENPV Model Inputs
    st.subheader("ENPV/CPR Model Parameters")
    NEW_TERM_YEARS = st.number_input("New Loan Term (years)", 
                                     value=30, min_value=10, max_value=30, step=5)
    INVEST_RATE = st.number_input("Investment Rate (%)", 
                                  value=4.0, min_value=0.0, max_value=20.0, step=0.1) / 100
    FINANCE_COSTS_IN_LOAN = st.checkbox("Roll closing costs into new loan", value=True)
    CPR = st.number_input("CPR for mortality (%)", 
                         value=11.5, min_value=0.0, max_value=50.0, step=0.5) / 100

# Helper functions
def lambertw0(x, tol=1e-14, max_iter=100):
    """Numerically approximate the principal branch of the Lambert W function W0(x)."""
    if x == 0:
        return 0.0
    
    if x > -0.1:
        w = x
    else:
        w = math.log(1 + x) if (1 + x) > 0 else -1.0
    
    for _ in range(max_iter):
        e_w = math.exp(w)
        f = w * e_w - x
        denom = e_w * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        if denom == 0:
            break
        w_new = w - f / denom
        if abs(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    
    return w

def alpha_pv(rho, lam, Gamma):
    """PV factor for points' tax shield."""
    d = rho + lam
    term1 = (1.0 - math.exp(-d * Gamma)) / (d * Gamma)
    term2 = lam * (
        (1.0 - math.exp(-d * Gamma)) / d
        - (1.0 - math.exp(-d * Gamma) * (1.0 + d * Gamma)) / (d * d * Gamma)
    )
    return term1 + term2

def effective_cost(M, p, F, tau, rho, lam, Gamma):
    """Calculate effective cost with tax adjustments."""
    pts = p * M
    kappa_pre = pts + F
    a = alpha_pv(rho, lam, Gamma)
    C_eff = kappa_pre / (1.0 - tau) - (tau / (1.0 - tau)) * pts * a
    return C_eff

def trigger_closed_form(i0, rho, lam, sigma, M, C_eff):
    """Calculate optimal refinancing trigger."""
    psi = math.sqrt(2.0 * (rho + lam)) / sigma
    phi = 1.0 + psi * (rho + lam) * (C_eff / M)
    W = lambertw0(-math.exp(-phi))
    
    x_star = -(phi + W) / psi
    x_PV = -(rho + lam) * (C_eff / M)
    i_star = i0 + x_star
    
    opt_rate_diff = abs(x_star) - abs(x_PV)
    opt_value_usd_approx = opt_rate_diff * M / (rho + lam)
    
    pv_savings_x_star = -x_star * M / (rho + lam)
    enpv_x_star = pv_savings_x_star - C_eff
    
    K = M * math.exp(psi * x_star) / (psi * (rho + lam))
    option_R0 = K
    
    return {
        "C_eff": C_eff,
        "psi": psi,
        "phi": phi,
        "W": W,
        "x_star": x_star,
        "x_PV": x_PV,
        "i_star": i_star,
        "opt_rate_diff": opt_rate_diff,
        "opt_value_usd_approx": opt_value_usd_approx,
        "pv_savings_x_star": pv_savings_x_star,
        "enpv_x_star": enpv_x_star,
        "option_R0": option_R0,
    }

def calculate_mortgage_payments(i_old, i_new, M, Gamma, financed_cost, rho, INVEST_RATE, NEW_TERM_YEARS):
    """Calculate mortgage payment comparisons and break-even analysis."""
    r_old = i_old / 12.0
    r_new = i_new / 12.0
    T = int(round(Gamma * 12))
    T_new = NEW_TERM_YEARS * 12
    
    def payment(principal, r, n):
        if r == 0.0:
            return principal / n
        denom = 1.0 - (1.0 + r) ** (-n)
        if denom == 0.0:
            return principal / n
        return principal * r / denom
    
    principal_old = M
    principal_new = M + financed_cost
    
    pmt_old = payment(principal_old, r_old, T)
    pmt_new = payment(principal_new, r_new, T_new)
    
    bal_old = principal_old
    bal_new = principal_new
    
    history = []
    breakeven_month = None
    
    for t in range(1, max(T, T_new) + 1):
        # Old loan
        if t <= T and bal_old > 0:
            interest_old = r_old * bal_old
            principal_old_pmt = pmt_old - interest_old
            bal_old = max(0.0, bal_old - principal_old_pmt)
        else:
            bal_old = 0.0
        
        # New loan
        if t <= T_new and bal_new > 0:
            interest_new = r_new * bal_new
            principal_new_pmt = pmt_new - interest_new
            bal_new = max(0.0, bal_new - principal_new_pmt)
        else:
            bal_new = 0.0
        
        history.append({
            "month": t,
            "bal_old": bal_old,
            "bal_new": bal_new,
            "pmt_old": pmt_old if t <= T else 0,
            "pmt_new": pmt_new if t <= T_new else 0
        })
        
        # Simple break-even check
        if breakeven_month is None and t <= T:
            total_saved = (pmt_old - pmt_new) * t
            if total_saved >= financed_cost:
                breakeven_month = t
    
    return history, pmt_old, pmt_new, breakeven_month

def calculate_npv_with_mortality(history, pmt_old, pmt_new, CPR, INVEST_RATE, Gamma, financed_cost):
    """Calculate NPV with mortality/prepayment probability."""
    SMM = 1 - (1 - CPR)**(1/12)
    
    # Calculate mortality distribution
    mortality = []
    for t in range(1, 360):
        m_t = (1 - SMM)**(t - 1) * SMM
        mortality.append(m_t)
    remaining = 1.0 - sum(mortality)
    mortality.append(remaining)
    
    # Calculate NPV for each month
    r_inv = INVEST_RATE / 12
    n_old = int(Gamma * 12)
    
    net_gain_pv = []
    opt1_sav = 0.0
    opt2_sav = 0.0
    
    for t in range(1, 361):
        if t <= len(history):
            rec = history[t-1]
            
            if t <= n_old:
                # Before original loan ends
                monthly_savings = rec["pmt_old"] - rec["pmt_new"]
                opt2_sav = opt2_sav * (1 + r_inv) + monthly_savings
                net_gain = opt2_sav - (rec["bal_new"] - rec["bal_old"])
            else:
                # After original loan ends
                opt1_sav = opt1_sav * (1 + r_inv) + pmt_old
                monthly_diff = pmt_old - rec["pmt_new"]
                opt2_sav = opt2_sav * (1 + r_inv) + monthly_diff
                net_gain = (opt2_sav - rec["bal_new"]) - opt1_sav
        else:
            # Beyond history, use last known values
            if t <= n_old:
                net_gain = net_gain_pv[-1] if net_gain_pv else 0
            else:
                opt1_sav = opt1_sav * (1 + r_inv) + pmt_old
                opt2_sav = opt2_sav * (1 + r_inv) + (pmt_old - pmt_new)
                net_gain = opt2_sav - opt1_sav
        
        # Discount to present value
        df = (1 + INVEST_RATE / 12) ** (-t)
        net_gain_pv.append(net_gain * df)
    
    # Calculate ENPV
    npv_times_mortality = [net_gain_pv[t] * mortality[t] for t in range(360)]
    ENPV = sum(npv_times_mortality)
    
    return net_gain_pv, mortality, ENPV

# Main calculations
C_eff = effective_cost(M, p, F, tau, rho, lam, Gamma)
res = trigger_closed_form(i0, rho, lam, sigma, M, C_eff)

# Display results in main area
tab1, tab2, tab3, tab4 = st.tabs(["📈 Option Model Results", "💰 Payment Analysis", "📊 ENPV Analysis", "📉 Interactive Charts"])

with tab1:
    st.header("Option Model Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Optimal Trigger Rate", f"{res['i_star']*100:.3f}%", 
                 f"{(res['i_star']-i0)*100:.3f}% from current")
        st.metric("NPV Trigger Rate", f"{(i0 + res['x_PV'])*100:.3f}%")
        st.metric("Effective Cost", f"${res['C_eff']:,.2f}")
    
    with col2:
        st.metric("Option Value (R₀)", f"${res['option_R0']:,.2f}")
        st.metric("Option Value (approx)", f"${res['opt_value_usd_approx']:,.2f}")
        st.metric("PV Savings at Trigger", f"${res['pv_savings_x_star']:,.2f}")
    
    with col3:
        st.metric("ENPV at Trigger", f"${res['enpv_x_star']:,.2f}")
        st.metric("Rate Drop Needed", f"{abs(res['x_star'])*100:.3f}%")
        st.metric("NPV Rule Drop", f"{abs(res['x_PV'])*100:.3f}%")
    
    # Explanation
    st.info("""
    **Key Insights:**
    - **Optimal Trigger Rate**: The mortgage rate at which you should refinance considering option value
    - **NPV Trigger Rate**: The rate at which refinancing has zero NPV (traditional rule)
    - **Option Value**: The value of waiting for potentially better rates vs. refinancing immediately
    """)

with tab2:
    st.header("Payment Comparison Analysis")
    
    # Calculate for a hypothetical new rate
    new_rate = st.slider("New Mortgage Rate (%)", 
                        min_value=1.0, 
                        max_value=float(i0*100), 
                        value=float(res['i_star']*100), 
                        step=0.1) / 100
    
    financed_cost = C_eff if FINANCE_COSTS_IN_LOAN else 0
    
    history, pmt_old, pmt_new, breakeven_month = calculate_mortgage_payments(
        i0, new_rate, M, Gamma, financed_cost, rho, INVEST_RATE, NEW_TERM_YEARS
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Payment", f"${pmt_old:,.2f}")
    with col2:
        st.metric("New Payment", f"${pmt_new:,.2f}")
    with col3:
        st.metric("Monthly Savings", f"${pmt_old - pmt_new:,.2f}")
    with col4:
        if breakeven_month:
            st.metric("Break-even Month", f"{breakeven_month}")
        else:
            st.metric("Break-even Month", "Never")
    
    # Create payment comparison chart
    months = [h["month"] for h in history[:360]]
    old_payments = [h["pmt_old"] for h in history[:360]]
    new_payments = [h["pmt_new"] for h in history[:360]]
    
    fig_payments = go.Figure()
    fig_payments.add_trace(go.Scatter(
        x=months, y=old_payments,
        name="Current Payment",
        line=dict(color='red', width=2)
    ))
    fig_payments.add_trace(go.Scatter(
        x=months, y=new_payments,
        name="New Payment",
        line=dict(color='green', width=2)
    ))
    
    fig_payments.update_layout(
        title="Monthly Payment Comparison",
        xaxis_title="Month",
        yaxis_title="Payment ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_payments, use_container_width=True)
    
    # Balance comparison
    old_balances = [h["bal_old"] for h in history[:360]]
    new_balances = [h["bal_new"] for h in history[:360]]
    
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(
        x=months, y=old_balances,
        name="Current Loan Balance",
        line=dict(color='red', width=2)
    ))
    fig_balance.add_trace(go.Scatter(
        x=months, y=new_balances,
        name="New Loan Balance",
        line=dict(color='green', width=2)
    ))
    
    fig_balance.update_layout(
        title="Loan Balance Over Time",
        xaxis_title="Month",
        yaxis_title="Balance ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_balance, use_container_width=True)

with tab3:
    st.header("Expected Net Present Value (ENPV) Analysis")
    
    # Calculate ENPV with mortality
    net_gain_pv, mortality, ENPV = calculate_npv_with_mortality(
        history, pmt_old, pmt_new, CPR, INVEST_RATE, Gamma, financed_cost
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ENPV", f"${ENPV:,.2f}")
    with col2:
        SMM = 1 - (1 - CPR)**(1/12)
        st.metric("Monthly Prepayment Rate (SMM)", f"{SMM*100:.3f}%")
    with col3:
        st.metric("Annual CPR", f"{CPR*100:.1f}%")
    
    # Create NPV chart
    months_npv = list(range(1, min(361, len(net_gain_pv) + 1)))
    npv_values = net_gain_pv[:360]
    
    fig_npv = go.Figure()
    fig_npv.add_trace(go.Scatter(
        x=months_npv, 
        y=npv_values,
        name="NPV of Refinancing",
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 200, 0.2)'
    ))
    
    fig_npv.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_npv.update_layout(
        title="Net Present Value of Refinancing Over Time",
        xaxis_title="Month",
        yaxis_title="NPV ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_npv, use_container_width=True)
    
    # Mortality distribution chart
    fig_mortality = go.Figure()
    fig_mortality.add_trace(go.Bar(
        x=list(range(1, min(61, len(mortality) + 1))),
        y=mortality[:60],
        name="Prepayment Probability",
        marker_color='purple'
    ))
    
    fig_mortality.update_layout(
        title="Monthly Prepayment Probability (First 60 Months)",
        xaxis_title="Month",
        yaxis_title="Probability",
        height=400
    )
    
    st.plotly_chart(fig_mortality, use_container_width=True)
    
    st.info("""
    **ENPV Interpretation:**
    - Positive ENPV indicates refinancing is beneficial on average
    - Accounts for the probability of prepayment/moving before loan maturity
    - Higher CPR reduces the expected benefit as you're less likely to keep the loan long-term
    """)

with tab4:
    st.header("Interactive Scenario Analysis")
    
    # Create rate scenario chart
    rate_scenarios = np.linspace(i0 * 0.5, i0, 50)
    enpvs = []
    
    for rate in rate_scenarios:
        temp_res = trigger_closed_form(rate, rho, lam, sigma, M, C_eff)
        enpvs.append(temp_res['enpv_x_star'])
    
    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(
        x=rate_scenarios * 100,
        y=enpvs,
        name="ENPV",
        line=dict(color='green', width=3)
    ))
    
    # Add markers for key points
    fig_scenario.add_trace(go.Scatter(
        x=[res['i_star'] * 100],
        y=[res['enpv_x_star']],
        mode='markers',
        name='Optimal Trigger',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig_scenario.add_trace(go.Scatter(
        x=[(i0 + res['x_PV']) * 100],
        y=[0],
        mode='markers',
        name='NPV Trigger',
        marker=dict(size=12, color='blue', symbol='diamond')
    ))
    
    fig_scenario.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_scenario.update_layout(
        title="ENPV vs. New Mortgage Rate",
        xaxis_title="New Mortgage Rate (%)",
        yaxis_title="ENPV ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)
    
    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    
    sensitivity_param = st.selectbox(
        "Select parameter to analyze:",
        ["CPR", "Volatility", "Discount Rate", "Tax Rate"]
    )
    
    if sensitivity_param == "CPR":
        param_range = np.linspace(0.05, 0.30, 20)
        param_label = "CPR"
        param_unit = "%"
        multiplier = 100
    elif sensitivity_param == "Volatility":
        param_range = np.linspace(0.001, 0.02, 20)
        param_label = "Volatility (σ)"
        param_unit = ""
        multiplier = 1
    elif sensitivity_param == "Discount Rate":
        param_range = np.linspace(0.01, 0.10, 20)
        param_label = "Discount Rate (ρ)"
        param_unit = "%"
        multiplier = 100
    else:  # Tax Rate
        param_range = np.linspace(0, 0.40, 20)
        param_label = "Tax Rate (τ)"
        param_unit = "%"
        multiplier = 100
    
    trigger_rates = []
    option_values = []
    
    for param_val in param_range:
        if sensitivity_param == "CPR":
            temp_res = trigger_closed_form(i0, rho, param_val, sigma, M, C_eff)
        elif sensitivity_param == "Volatility":
            temp_res = trigger_closed_form(i0, rho, lam, param_val, M, C_eff)
        elif sensitivity_param == "Discount Rate":
            temp_res = trigger_closed_form(i0, param_val, lam, sigma, M, C_eff)
        else:  # Tax Rate
            temp_C_eff = effective_cost(M, p, F, param_val, rho, lam, Gamma)
            temp_res = trigger_closed_form(i0, rho, lam, sigma, M, temp_C_eff)
        
        trigger_rates.append(temp_res['i_star'] * 100)
        option_values.append(temp_res['option_R0'])
    
    # Create subplot with two y-axes
    fig_sensitivity = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_sensitivity.add_trace(
        go.Scatter(
            x=param_range * multiplier,
            y=trigger_rates,
            name="Trigger Rate",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    fig_sensitivity.add_trace(
        go.Scatter(
            x=param_range * multiplier,
            y=option_values,
            name="Option Value",
            line=dict(color='red', width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    fig_sensitivity.update_xaxes(title_text=f"{param_label} {param_unit}")
    fig_sensitivity.update_yaxes(title_text="Trigger Rate (%)", secondary_y=False)
    fig_sensitivity.update_yaxes(title_text="Option Value ($)", secondary_y=True)
    
    fig_sensitivity.update_layout(
        title=f"Sensitivity to {param_label}",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### About This Calculator
This advanced mortgage refinancing calculator implements sophisticated financial models to help you make optimal refinancing decisions.
It considers not just immediate savings but also the option value of waiting for potentially better rates in the future.

**Key Features:**
- Option pricing model for optimal trigger rates
- Expected Net Present Value with prepayment probabilities
- Interactive charts and sensitivity analysis
- Tax-adjusted cost calculations
""")
