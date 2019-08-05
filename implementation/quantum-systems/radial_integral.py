def radial_integral(r_p, r_q):
    r = sympy.Symbol("r")

    return sympy.integrate(
        r ** 2 * r_p(r).conjugate() * r_q(r),
        (r, 0, sympy.oo),
    )
