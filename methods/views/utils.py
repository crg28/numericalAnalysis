# methods/views/utils.py
from __future__ import annotations
import io
import importlib
import inspect
from contextlib import redirect_stdout

from sympy import Matrix, sympify, lambdify, symbols


# -----------------------------
# Symbolic compiler f(x)
# -----------------------------
_x = symbols("x")

def compile_fx(expr_text: str):
    """Return (callable f(x), sympy_expr) from a user string."""
    expr = sympify(expr_text, convert_xor=True)
    f = lambdify(_x, expr, "numpy")
    return f, expr



# -----------------------------
# Session helpers
# -----------------------------
def load_last(request, kind: str) -> dict:
    bucket = request.session.get("last_inputs", {})
    return bucket.get(kind, {})


def save_last(request, kind: str, cleaned: dict):
    bucket = request.session.get("last_inputs", {})
    bucket[kind] = {k: cleaned.get(k) for k in cleaned}
    request.session["last_inputs"] = bucket


# -----------------------------
# Invoke ROOT-finding algorithms
# -----------------------------
_ROOT_FUN_CANDIDATES = (
    "run", "solve", "algorithm", "main", "execute",
    "bisection", "biseccion",
    "false_position", "regla_falsa",
    "fixed_point", "punto_fijo",
    "incremental_search", "busqueda_incremental",
    "newton", "secant",
    "multiple_roots", "roots_multiplicity",
)

def _pick_callable(mod):
    for name in _ROOT_FUN_CANDIDATES:
        if hasattr(mod, name):
            return getattr(mod, name), name
    return None, None

def invoke_root_algorithm(kind: str, expr_text: str, f_lambda, params: dict) -> str | None:
    """
    Import methods/algorithms/<kind>.py and attempt multiple signatures.
    Captures EXACT stdout from the underlying algorithm.
    """
    module_name = kind.replace("-", "_")

    # Mapeo especial de nombre de módulo para raíces múltiples
    if kind == "multiple_roots":
        module_name = "roots_multiplicity"

    try:
        mod = importlib.import_module(f"methods.algorithms.{module_name}")
    except Exception:
        return None

    # ---- CASO ESPECIAL: múltiples raíces ----
    if kind == "multiple_roots":
        if not hasattr(mod, "roots_multiplicity"):
            return None
        fn = getattr(mod, "roots_multiplicity")

        x0 = params.get("x0")
        tol = params.get("tol", 1e-6)
        n   = params.get("max_iter", 50)
        df_str  = params.get("df_str", "")
        d2f_str = params.get("d2f_str", "")

        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(x0, expr_text, df_str, d2f_str, tol=tol, max_iter=n)
        out = buf.getvalue()
        return out if out else None
    # -----------------------------------------

    # ---- Resto de métodos (bisección, fija, newton, secante, etc.) ----
    # Elegir la función "principal" del módulo (por convención)
    fn, _ = _pick_callable(mod)
    if not fn:
        return None

    # Extraer parámetros comunes desde params
    a      = params.get("a")
    b      = params.get("b")
    x0     = params.get("x0")
    x1     = params.get("x1")
    delta  = params.get("delta")
    tol    = params.get("tol", 1e-6)
    n      = params.get("max_iter", 50)
    df_str = params.get("df_str")  # 🔹 NUEVO: para métodos que acepten df_str (p.ej. Newton)

    sig = inspect.signature(fn)
    names = list(sig.parameters.keys())

    call_variants = []

    # Keyword base: siempre intentamos pasar estos si la firma los acepta
    kw = {"tol": tol, "max_iter": n}
    if a is not None:      kw["a"] = a
    if b is not None:      kw["b"] = b
    if x0 is not None:     kw["x0"] = x0
    if x1 is not None:     kw["x1"] = x1
    if delta is not None:  kw["delta"] = delta
    if df_str is not None: kw["df_str"] = df_str  # 🔹 NUEVO: solo lo recibirá quien tenga df_str en su firma

    # Variantes keyword donde la función se pasa como callable
    # Aquí añadimos también 'g' para punto fijo
    for key in ("f", "func", "function", "g"):
        call_variants.append(({}, {**kw, key: f_lambda}))

    # Variantes keyword donde la función se pasa como string
    # Y añadimos 'g_str' para las variantes con string (punto fijo)
    for key in ("f_str", "fx", "expr", "expr_str", "g_str"):
        call_variants.append(({}, {**kw, key: expr_text}))

    # Variantes posicionales
    pos_args_sets = []

    # Patrones típicos [a,b]
    if a is not None and b is not None:
        pos_args_sets.append([expr_text, a, b, tol, n])
        pos_args_sets.append([f_lambda,  a, b, tol, n])

    # Patrones con x0, x1 (secante)
    if x0 is not None and x1 is not None:
        pos_args_sets.append([expr_text, x0, x1, tol, n])
        pos_args_sets.append([f_lambda,  x0, x1, tol, n])

    # Patrones con x0, delta (búsqueda incremental)
    if x0 is not None and delta is not None:
        pos_args_sets.append([expr_text, x0, delta, n])
        pos_args_sets.append([f_lambda,  x0, delta, n])

    # 🔸 Patrón típico de punto fijo (g, x0, tol, n),
    # cuando no hay a, b, x1, delta
    if x0 is not None and a is None and b is None and x1 is None and delta is None:
        pos_args_sets.append([expr_text, x0, tol, n])
        pos_args_sets.append([f_lambda,  x0, tol, n])

    for args in pos_args_sets:
        call_variants.append(({"_pos": args}, {}))

    # Ejecutar y capturar stdout
    buf = io.StringIO()
    for args_dict, kwargs in call_variants:
        try:
            with redirect_stdout(buf):
                if "_pos" in args_dict:
                    fn(*args_dict["_pos"])
                else:
                    # Filtrar solo los kwargs que la función realmente acepta
                    filtered = {k: v for k, v in kwargs.items() if k in names}
                    fn(**filtered)
            break  # si una llamada funciona, dejamos de probar variantes
        except Exception:
            # probamos la siguiente variante
            pass

    out = buf.getvalue()
    return out if out else None
