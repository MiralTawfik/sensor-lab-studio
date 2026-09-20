"""LaTeX symbols for the output / input quantities used in equations."""
SYMBOLS = {
    "V_out": r"V_{\mathrm{out}}", "dV": r"\Delta V", "d_lambda": r"\Delta\lambda", "d_lambda_B": r"\Delta\lambda_B",
    "d_theta": r"\Delta\theta", "dn_s": r"\Delta n_s", "eps": r"\varepsilon", "a": "a", "P": "P", "F": "F",
}


def latex(sym: str) -> str:
    return SYMBOLS.get(sym, sym)


PLAIN = {
    "V_out": "V_out", "dV": "ΔV", "d_lambda": "Δλ", "d_lambda_B": "Δλ_B", "d_theta": "Δθ", "dn_s": "Δn_s", "eps": "ε",
}


def plain(sym: str) -> str:
    return PLAIN.get(sym, sym)
