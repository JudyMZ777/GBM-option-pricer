
import numpy as np
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="GBM Option Pricer", page_icon="💹", layout="centered")

# note
ITO_TEXT = r"""
**Itô's formula (1D):**  
If $X_t$ is an Itô process with
$$\mathrm{d}X_t = a(t,X_t)\,\mathrm{d}t + b(t,X_t)\,\mathrm{d}W_t$$
and $f \in C^{1,2}$, then
$$\mathrm{d}f(t,X_t) = \left( f_t + a f_x + \tfrac{1}{2} b^2 f_{xx} \right) \mathrm{d}t + b f_x \mathrm{d}W_t.$$  

**GBM SDE under risk-neutral measure:**  
$$ \mathrm{d}S_t = (r - q) S_t \, \mathrm{d}t + \sigma S_t \, \mathrm{d}W_t, $$
$$ S_T = S_0 \, \exp\!\left((r-q-\tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\right), \quad Z\sim N(0,1). $$  

**Black–Scholes prices (with continuous dividend yield $q$):**  
$$ d_1 = \frac{\ln(S_0/K) + (r-q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}, $$
$$ C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2), \quad P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1). $$
"""

# pricing function

def _validate_inputs(S: float, K: float, r: float, sigma: float, T: float):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive.")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative.")


def black_scholes_prices(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0):
    _validate_inputs(S, K, r, sigma, T)
    if T == 0 or sigma == 0:
        call = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0.0)
        put = max(K * np.exp(-r*T) - S * np.exp(-q*T), 0.0)
        return float(call), float(put)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return float(call), float(put)


def monte_carlo_prices(S, K, r, sigma, T, q=0.0, n_paths=100_000, antithetic=True, seed=42):
    _validate_inputs(S, K, r, sigma, T)
    rng = np.random.default_rng(seed)
    n = max(1, int(n_paths))
    if antithetic:
        m = (n + 1) // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])[:n]
    else:
        Z = rng.standard_normal(n)

    drift = (r - q - 0.5 * sigma**2) * T
    diff = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diff)

    disc_r = np.exp(-r * T)

    call_payoffs = np.maximum(S_T - K, 0.0)
    put_payoffs = np.maximum(K - S_T, 0.0)

    call = disc_r * np.mean(call_payoffs)
    put = disc_r * np.mean(put_payoffs)

    call_se = disc_r * np.std(call_payoffs, ddof=1) / np.sqrt(n)
    put_se = disc_r * np.std(put_payoffs, ddof=1) / np.sqrt(n)

    return float(call), float(put), float(call_se), float(put_se)

# UI
st.title("💹 GBM Option Pricer")
st.caption("Switch between analytical Black–Scholes and Monte Carlo simulation. Study notes included.")

with st.sidebar:
    st.header("Inputs")
    S = st.number_input("Spot price S₀", min_value=0.0, value=100.0, step=1.0)
    K = st.number_input("Strike K", min_value=0.0, value=100.0, step=1.0)
    T = st.number_input("Maturity T (years)", min_value=0.0, value=1.0, step=0.1, format="%.4f")
    r_pct = st.number_input("Risk-free rate r (%)", value=5.0, step=0.25, format="%.4f")
    q_pct = st.number_input("Dividend yield q (%)", value=0.0, step=0.25, format="%.4f")
    sigma_pct = st.number_input("Volatility σ (%)", min_value=0.0, value=20.0, step=1.0, format="%.4f")

    r = r_pct / 100.0
    q = q_pct / 100.0
    sigma = sigma_pct / 100.0

    method = st.selectbox("Method", ["Analytical (Black–Scholes)", "Simulation (Monte Carlo)"])

    if method == "Simulation (Monte Carlo)":
        n_paths = st.number_input("Number of paths", min_value=1, value=100_000, step=10_000)
        antithetic = st.checkbox("Antithetic variates", value=True)
        seed_opt = st.checkbox("Set random seed", value=True)
        seed = st.number_input("Seed", value=42, step=1) if seed_opt else None
    else:
        n_paths = None
        antithetic = None
        seed = None

    show_notes = st.checkbox("Show study notes (Itô, GBM, BSM)", value=False)

try:
    if method == "Analytical (Black–Scholes)":
        call, put = black_scholes_prices(S, K, r, sigma, T, q=q)
        st.subheader("Analytical prices")
        col1, col2 = st.columns(2)
        col1.metric("Call", f"{call:,.6f}")
        col2.metric("Put", f"{put:,.6f}")
    else:
        call, put, call_se, put_se = monte_carlo_prices(S, K, r, sigma, T, q=q, n_paths=int(n_paths), antithetic=bool(antithetic), seed=None if seed is None else int(seed))
        st.subheader("Monte Carlo prices")
        col1, col2 = st.columns(2)
        col1.metric("Call", f"{call:,.6f}", help=f"Std. error ≈ {call_se:,.6f}")
        col2.metric("Put", f"{put:,.6f}", help=f"Std. error ≈ {put_se:,.6f}")

        bsm_call, bsm_put = black_scholes_prices(S, K, r, sigma, T, q=q)
        st.caption(f"Black–Scholes (closed-form) benchmark → Call: {bsm_call:,.6f}, Put: {bsm_put:,.6f}")

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    parity_lhs = (call - put)
    parity_rhs = S * disc_q - K * disc_r
    st.markdown("**Put–call parity check:**  ")
    st.latex(r"C - P = S e^{-qT} - K e^{-rT}")
    st.write(f"Computed: {parity_lhs:,.6f}  |  RHS: {parity_rhs:,.6f}")

except Exception as e:
    st.error(f"Input error: {e}")

if show_notes:
    st.divider()
    st.markdown("## Study notes")
    st.markdown(ITO_TEXT)
    st.markdown(r"""
**Analytical vs Simulation (intuition):**
- *Analytical* (Black–Scholes–Merton) gives closed-form prices under GBM and constant $\sigma$, $r$, $q$.
- *Simulation* generates many scenarios of $S_T$ from the GBM distribution and averages discounted payoffs. Accuracy improves as $N$ grows (error $\sim 1/\sqrt{N}$).
- Under plain GBM, Monte Carlo should converge to Black–Scholes; differences reflect finite-sample error.

**Tips:**
- Scale inputs to yearly units (T in years, $r$, $q$, $\sigma$ as annualized decimals).
- For path-dependent payoffs (barriers, Asians) you need time discretization; here we use the exact terminal law.
    """)
