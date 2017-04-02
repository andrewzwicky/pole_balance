from sympy import *
import re
t = symbols('t')
init_printing(use_unicode=True)

theta_re = re.compile(r"\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")
deriv_re = re.compile(r"\\frac{d}{d t}\s+\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")
dderiv_re = re.compile(r"\\frac{d\^\{2}}{d t\^\{2}}\s+\\operatorname{T_{(\d+)}}{\\left \(t \\right \)}")

def pformat(s):
    s = dderiv_re.sub(r"\\ddot{\\theta_\g<1>}", s)
    s = deriv_re.sub(r"\\dot{\\theta_\g<1>}", s)
    s = theta_re.sub(r"\\theta_\g<1>", s)
    return "=& " + s + r"\\"

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
M1 = 1.0  # mass of pendulum 1 in kg
I1 = (M1 * L1 * L1) / 3  # moment of inertia of pendulum 1, fixed at end
L2 = 1.0  # length of pendulum 2 in m
M2 = 1.0  # mass of pendulum 2 in kg
I2 = (M2 * L2 * L2) / 3  # moment of inertia of pendulum 2, fixed at end
R1 = L1/2
R2 = L2/2

T1 = "T1(t)"
T2 = "T2(t)"

x1 = "r_1*cos({0})".format(T1)
y1 = "r_1*sin({0})".format(T1)
x2 = "l_1*cos({0}) + r_2*cos({0}+{1})".format(T1,T2)
y2 = "l_1*sin({0}) + r_2*sin({0}+{1})".format(T1,T2)

x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

T1_dot = diff(T1, t)
T2_dot = diff(T2, t)

K = "(m_1*(({0})^2+({1})^2) + m_2*(({2})^2+({3})^2) + I_1*({4})^2 + I_2*({5})^2)/2".format(x1_dot, y1_dot, x2_dot, y2_dot, T1_dot, T2_dot)
U = "m_1*g*l_1*sin({0})+m_2*g*(l_1*sin({0})+l_2*sin({0}+{1}))".format(T1, T2)
L = "{0} - {1}".format(K, U)


print(pformat(latex(factor(L))))
print(pformat(latex(factor(diff(L, T1)))))
print(pformat(latex(factor(diff(L, T2)))))
print(pformat(latex(factor(diff(diff(L, T1_dot),t)))))
print(pformat(latex(collect(diff(diff(L, T2_dot),t), T2_dot))))