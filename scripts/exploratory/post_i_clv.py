# POST-HOC, không dùng để đổi kết luận P1–P3
"""BG/NBD (Fader-Hardie-Lee 2005) và Pareto/NBD (Schmittlein-Morrison-Colombo 1987) tự cài bằng scipy
(thư viện lifetimes chưa có trong môi trường; không cài thêm). Đơn vị thời gian: tuần.
Dữ liệu đầu vào cho mỗi người: x = số lần mua lặp (= số ngày mua - 1), t_x = ngày mua cuối - ngày mua đầu,
T = mốc cắt - ngày mua đầu tiên quan sát được.
P(không mua trong (T, T+tau] | x, t_x, T) = L(x, t_x, T+tau) / L(x, t_x, T)   (cùng x, t_x), với L là hàm hợp lý.
Chạy trực tiếp file này để kiểm chứng bằng mô phỏng (khôi phục tham số + hiệu chuẩn)."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1, logsumexp


def bgnbd_ll(p, x, tx, T):
    r, al, a, b = p
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(al)
    A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    A3 = -(r + x) * np.log(al + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        A4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(al + tx), -np.inf)
    return A1 + A2 + np.logaddexp(A3, A4)


def _log_2f1_euler(a, b, c, z):
    """log 2F1(a,b;c;z) với c = a + 1, qua biến đổi Euler: (1-z)^(c-a-b) 2F1(c-a, c-b; c; z), c-a = 1.
    Dạng gốc tràn số (NaN/inf) khi x lớn (x ~ 900); dạng này bị chặn. Lưu ý: log-lik dương là bình thường vì L là mật độ."""
    return (c - a - b) * np.log1p(-z) + np.log(hyp2f1(1.0, c - b, c, z))


def pareto_ll(p, x, tx, T):
    """Dạng đóng theo Fader & Hardie (2005), "A Note on Deriving the Pareto/NBD Model": hai nhánh theo alpha, beta."""
    r, al, s, be = p
    rsf = r + s + x
    if al >= be:
        mx, b2, d = al, s + 1.0, al - be
    else:
        mx, b2, d = be, r + x, be - al
    a1 = _log_2f1_euler(rsf, b2, rsf + 1, d / (mx + tx)) - rsf * np.log(mx + tx)
    a2 = _log_2f1_euler(rsf, b2, rsf + 1, d / (mx + T)) - rsf * np.log(mx + T)
    # A0 = F(t_x)/(mx+t_x)^rsf - F(T)/(mx+T)^rsf >= 0 (bằng 0 khi t_x = T)
    with np.errstate(divide="ignore", invalid="ignore"):
        logA0 = np.where(a1 > a2, a1 + np.log1p(-np.exp(np.minimum(a2 - a1, 0.0))), -np.inf)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(al) + s * np.log(be)
    A2 = np.logaddexp(-(r + x) * np.log(al + T) - s * np.log(be + T), np.log(s) - np.log(rsf) + logA0)
    return A1 + A2


def _large_x_check():
    """x rất lớn như dữ liệu thật (x ~ 950): tích phân số của L/L_dạng_đóng phải bằng 1 (tính trong log để khỏi tràn)."""
    from scipy.integrate import dblquad
    from scipy.stats import gamma as Ga
    worst = 0.0
    for r, al, s, be in [(1.15, 1.84, 0.046, 275.0), (1.15, 300.0, 0.5, 2.0)]:
        for x, tx, T in [(863, 207.14, 207.15), (957, 208.29, 208.43), (952, 200.0, 221.5), (500, 100.0, 200.0)]:
            c = float(pareto_ll((r, al, s, be), np.array([float(x)]), np.array([tx]), np.array([T]))[0])

            def f(mu, lam):
                l1 = x * np.log(lam) + np.log(mu) - np.log(lam + mu) - (lam + mu) * tx
                l2 = (x + 1) * np.log(lam) - np.log(lam + mu) - (lam + mu) * T
                return np.exp(np.logaddexp(l1, l2) + Ga.logpdf(lam, r, scale=1 / al) + Ga.logpdf(mu, s, scale=1 / be) - c)
            pk = (r + x) / (al + tx)   # đỉnh hậu nghiệm của lambda (khoảng Gamma(r+x, al+t) với t trong [t_x, T])
            val = dblquad(f, pk * 0.5, pk * 1.6, 0, np.inf, epsabs=1e-7, epsrel=1e-7)[0]
            worst = max(worst, abs(val - 1))
            print(f"   pareto large-x a={al} b={be} x={x} T={T}: integral/closed={val:.8f} ll={c:.2f}")
    return worst


def _numeric_check():
    """So dạng đóng với tích phân số trực tiếp của hàm hợp lý có điều kiện trên phân phối trộn Gamma/Beta."""
    from scipy.integrate import dblquad
    from scipy.stats import beta as Be, gamma as Ga
    pts = [(0, 0.0, 10.0), (0, 0.0, 0.0), (3, 7.0, 20.0), (3, 20.0, 20.0), (12, 40.0, 100.0)]
    worst = 0.0
    for r, al, s, be in [(0.8, 8.0, 0.6, 30.0), (0.8, 30.0, 0.6, 8.0), (1.5, 12.0, 1.2, 12.0)]:
        for x, tx, T in pts:
            def f(mu, lam):
                L = (lam ** x * mu / (lam + mu)) * np.exp(-(lam + mu) * tx) + (lam ** (x + 1) / (lam + mu)) * np.exp(-(lam + mu) * T)
                return L * Ga.pdf(lam, r, scale=1 / al) * Ga.pdf(mu, s, scale=1 / be)
            num = dblquad(f, 0, np.inf, 0, np.inf, epsabs=1e-24, epsrel=1e-8)[0]
            cf = float(np.exp(pareto_ll((r, al, s, be), np.array([float(x)]), np.array([tx]), np.array([T]))[0]))
            rel = abs(cf - num) / max(num, 1e-300)
            worst = max(worst, rel)
            print(f"   pareto a={al} b={be} x={x} tx={tx} T={T}: closed={cf:.6e} numeric={num:.6e} rel={rel:.1e}")
    for r, al, a, b in [(0.8, 8.0, 0.7, 2.0)]:
        for x, tx, T in pts:
            def g(p, lam):
                L = (1 - p) ** x * lam ** x * np.exp(-lam * T)
                if x > 0:
                    L += p * (1 - p) ** (x - 1) * lam ** x * np.exp(-lam * tx)
                return L * Ga.pdf(lam, r, scale=1 / al) * Be.pdf(p, a, b)
            num = dblquad(g, 0, np.inf, 0, 1, epsabs=1e-24, epsrel=1e-8)[0]
            cf = float(np.exp(bgnbd_ll((r, al, a, b), np.array([float(x)]), np.array([tx]), np.array([T]))[0]))
            rel = abs(cf - num) / max(num, 1e-300)
            worst = max(worst, rel)
            print(f"   bgnbd x={x} tx={tx} T={T}: closed={cf:.6e} numeric={num:.6e} rel={rel:.1e}")
    return worst


LL = {"bgnbd": bgnbd_ll, "pareto": pareto_ll}
START = {"bgnbd": [0.5, 10.0, 0.8, 2.5], "pareto": [0.5, 10.0, 0.5, 10.0]}


def fit(kind, x, tx, T, maxiter=300):
    """Máy ước lượng hợp lý cực đại trên log-tham số (Nelder-Mead rồi L-BFGS-B). Trả về (tham số, log-lik trung bình)."""
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    ll = LL[kind]

    def nll(lp):
        v = ll(np.exp(lp), x, tx, T)
        v = np.where(np.isfinite(v), v, -1e6)
        return -float(np.mean(v))

    best = None
    for start in (START[kind], [1.0, 20.0, 1.0, 5.0] if kind == "bgnbd" else [1.0, 20.0, 1.0, 20.0]):
        r0 = minimize(nll, np.log(start), method="Nelder-Mead", options=dict(maxiter=maxiter * 4, xatol=1e-5, fatol=1e-9))
        r1 = minimize(nll, r0.x, method="L-BFGS-B", bounds=[(-9, 9)] * 4, options=dict(maxiter=maxiter))
        r = r1 if r1.fun <= r0.fun else r0
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), -best.fun


def p_no_purchase(kind, params, x, tx, T, tau):
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    ll = LL[kind]
    lp = ll(params, x, tx, T + tau) - ll(params, x, tx, T)
    if not np.isfinite(lp).all():
        raise FloatingPointError(f"{kind}: {int((~np.isfinite(lp)).sum())} log-lik ratio không hữu hạn")
    if lp.max() > 1e-8:
        raise FloatingPointError(f"{kind}: P0 > 1 (max log = {lp.max()})")
    return np.clip(np.exp(lp), 0.0, 1.0)


def simulate(kind, params, n, T, seed=0, tau=13.0):
    rng = np.random.default_rng(seed)
    if kind == "bgnbd":
        r, al, a, b = params
        lam = rng.gamma(r, 1 / al, n)
        p_drop = rng.beta(a, b, n)
        x = np.zeros(n); tx = np.zeros(n); future = np.zeros(n)
        for i in range(n):
            t, alive = 0.0, True
            while True:
                t += rng.exponential(1 / lam[i])
                # drop-out is decided right after each purchase
                if t > T + tau:
                    break
                if t <= T:
                    x[i] += 1; tx[i] = t
                else:
                    future[i] += 1
                if rng.random() < p_drop[i]:
                    break
        return x, tx, np.full(n, T), future
    r, al, s, be = params
    lam = rng.gamma(r, 1 / al, n)
    mu = rng.gamma(s, 1 / be, n)
    life = rng.exponential(1 / mu)
    x = np.zeros(n); tx = np.zeros(n); future = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > min(life[i], T + tau):
                break
            if t <= T:
                x[i] += 1; tx[i] = t
            else:
                future[i] += 1
    return x, tx, np.full(n, T), future


if __name__ == "__main__":
    import time
    w = _numeric_check()
    print("worst relative error closed form vs numeric integral:", w)
    assert w < 1e-3
    w2 = _large_x_check()
    print("worst |integral/closed - 1| at large x:", w2)
    assert w2 < 1e-3
    for kind, true in (("bgnbd", [0.8, 8.0, 0.7, 2.0]), ("pareto", [0.8, 8.0, 0.6, 30.0]), ("pareto", [0.8, 30.0, 0.6, 8.0])):
        t0 = time.time()
        x, tx, T, fut = simulate(kind, true, 3000, 100.0, seed=1)
        # the fitting data must not include the future
        est, ll = fit(kind, x, tx, T)
        p0 = p_no_purchase(kind, est, x, tx, T, 13.0)
        print(kind, "true", true, "est", np.round(est, 3), "ll", round(ll, 4), f"{time.time() - t0:.1f}s")
        print("   mean predicted P0 = %.4f vs simulated frequency of no purchase = %.4f" % (p0.mean(), (fut == 0).mean()))
        q = np.quantile(p0, [0, .25, .5, .75, 1])
        for lo, hi in zip(q[:-1], q[1:]):
            m = (p0 >= lo) & (p0 <= hi)
            print("   bin pred %.3f  obs %.3f  n=%d" % (p0[m].mean(), (fut[m] == 0).mean(), m.sum()))
