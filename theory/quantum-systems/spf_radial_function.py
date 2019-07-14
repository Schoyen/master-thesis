def spf_radial_function(n, m, mass, omega):
    a = sympy.Float(bohr_radius(mass, omega))

    radial_function = (
        lambda r: (a * r) ** abs(m)
        * sympy.assoc_laguerre(n, abs(m), a ** 2 * r ** 2)
        * sympy.exp(-a ** 2 * r ** 2 / 2.0)
    )

    return radial_function
