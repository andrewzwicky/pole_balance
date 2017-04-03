from contextlib import contextmanager
from sympy import *
from pprint import pprint
import re

t, r_1, r_2, l_1, l_2, g, m_1, I_1, m_2, I_2, f_1, f_2, b_1, b_2 = symbols('t r_1 r_2 l_1 l_2 g m_1 I_1 m_2 I_2 f_1 f_2 b_1 b_2')
T1 = Function('T1')(t)
T2 = Function('T2')(t)
T1_dot = Derivative(T1, t)
T2_dot = Derivative(T2, t)
T1_ddot = Derivative(T1_dot, t)
T2_ddot = Derivative(T2_dot, t)


init_printing(use_unicode=True)

theta_re = re.compile(r"\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")
deriv_re = re.compile(r"\\frac{d}{d t}\s+\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")
dderiv_re = re.compile(r"\\frac{d\^\{2}}{d t\^\{2}}\s+\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")


def pformat(equation, prefix=None):
    if prefix is None:
        prefix = ""
    equation = latex(equation)
    equation = dderiv_re.sub(r"\\ddot{\\theta_\g<1>}", equation)
    equation = deriv_re.sub(r"\\dot{\\theta_\g<1>}", equation)
    equation = theta_re.sub(r"\\theta_\g<1>", equation)
    return prefix + "=& " + equation + r" & \\"


@contextmanager
def flalign_tag():
    print(r"\begin{flalign}")
    yield
    print(r"\end{flalign}")
    print(r"")


def print_lagrange(index, lagrange, theta, theta_dot, time):

    a = simplify(diff(lagrange, theta_dot))
    b = simplify(diff(a, time))
    c = simplify(diff(lagrange, theta))
    d = simplify(b - c)

    d_dot = r"\frac{{\partial{{L}}}}{{\partial{{\dot{{\theta_{index}}}}}}} {equation}"
    d_dot_t = r"\frac{{d}}{{dt}}\left(\frac{{\partial{{L}}}}{{\partial{{\dot{{\theta_{index}}}}}}}\right) {equation}"
    d_theta = r"\frac{{\partial{{L}}}}{{\partial{{\theta_{index}}}}} {equation}"
    combined = r"\frac{{d}}{{dt}}\left(\frac{{\partial{{L}}}}{{\partial{{\dot{{\theta_{index}}}}}}}\right) - \frac{{\partial{{L}}}}{{\partial{{\theta_{index}}}}} {equation}"

    print(d_dot.format(index=index, equation=pformat(a)))
    print(d_dot_t.format(index=index, equation=pformat(b)))
    print(d_theta.format(index=index, equation=pformat(c)))
    print(combined.format(index=index, equation=pformat(d)))

    return d

x1 = r_1 * cos(T1)
y1 = r_1 * sin(T1)
x2 = l_1 * cos(T1) + r_2 * cos(T1 + T2)
y2 = l_1 * sin(T1) + r_2 * sin(T1 + T2)

x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

K = ((m_1 * (x1_dot ** 2 + y1_dot ** 2) + m_2*(x2_dot ** 2 + y2_dot ** 2) + I_1 * T1_dot ** 2 + I_2 * T2_dot ** 2) / 2)
U = m_1 * g * l_1 * sin(T1) + m_2 * g * (l_1 * sin(T1) + l_2 * sin(T1 + T2))
L = K - U

with flalign_tag():
    print(pformat(simplify(x1), prefix="x_1"))
    print(pformat(simplify(y1), prefix="y_1"))
    print(pformat(simplify(x2), prefix="x_2"))
    print(pformat(simplify(y2), prefix="y_2"))
    print(pformat(simplify(x1_dot), prefix="\dot{x_1}"))
    print(pformat(simplify(y1_dot), prefix="\dot{y_1}"))
    print(pformat(simplify(x2_dot), prefix="\dot{x_2}"))
    print(pformat(simplify(y2_dot), prefix="\dot{y_2}"))

with flalign_tag():
    print(r"K =& \tfrac{1}{2}I_1\omega_1^2 + \tfrac{1}{2}m_1v_1^2 + \tfrac{1}{2}I_2\omega_2^2 + \tfrac{1}{2}m_2v_2^2 & \\")
    print(pformat(simplify(K)))

with flalign_tag():
    print(r"U =& m_1gl_1\sin(\theta_1) + m_2g(l_1\sin(\theta_1)+l_2\sin(\theta_1+\theta_2)) & \\")
    print(pformat(simplify(U)))

with flalign_tag():
    print(r"L =& K -U & \\")
    print(pformat(simplify(L)))

with flalign_tag():
    L1 = print_lagrange(1, L, T1, T1_dot, t)
    L2 = print_lagrange(2, L, T2, T2_dot, t)

Lagrange1 = L1 - (f_1 + b_1 * T1_dot)
Lagrange2 = L2 - (f_2 + b_2 * T2_dot)
