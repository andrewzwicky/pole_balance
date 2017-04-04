import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import sympy
from sympy import symbols, diff, Function, Derivative, poly, simplify, Matrix, init_printing, solve_linear_system, latex
from sympy.physics.vector import vlatex
from IPython.display import display, Math, Latex

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
M1 = 1.0  # mass of pendulum 1 in kg
I1 = (M1 * L1 * L1) / 12  # moment of inertia of pendulum 1, fixed at end
L2 = 1.0  # length of pendulum 2 in m
M2 = 1.0  # mass of pendulum 2 in kg
I2 = (M2 * L2 * L2) / 12  # moment of inertia of pendulum 2, fixed at end
R1 = L1 / 2
R2 = L2 / 2
B1 = 0
B2 = 0
T1 = 0
T2 = 0

init_printing(latex_printer=vlatex)

g, t = symbols('g t')
r_1, l_1, m_1, I_1, b_1 = symbols('r_1 l_1 m_1 I_1 b_1')
r_2, l_2, m_2, I_2, b_2 = symbols('r_2 l_2 m_2 I_2 b_2')
tau_1, tau_2 = symbols('tau_1 tau_2')

theta = Function('theta')(t)
phi = Function('phi')(t)
theta_dot = Derivative(theta, t)
phi_dot = Derivative(phi, t)
theta_ddot = Derivative(theta, t, 2)
phi_ddot = Derivative(phi, t, 2)

x1 = r_1 * sympy.cos(theta)
y1 = r_1 * sympy.sin(theta)
x2 = l_1 * sympy.cos(theta) + r_2 * sympy.cos(theta + phi)
y2 = l_1 * sympy.sin(theta) + r_2 * sympy.sin(theta + phi)

x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

K = (m_1 * (x1_dot ** 2 + y1_dot ** 2)
     + m_2 * (x2_dot ** 2 + y2_dot ** 2)
     + I_1 * theta_dot ** 2
     + I_2 * (theta_dot + phi_dot) ** 2) / 2
U = (m_1 * g * r_1 * sympy.sin(theta)) + (m_2 * g * (l_1 * sympy.sin(theta) + r_2 * sympy.sin(theta + phi)))
L = K - U

L_1 = diff(L, theta_dot, t) - diff(L, theta) - (tau_1 - b_1 * theta_dot)
L_2 = diff(L, phi_dot, t) - diff(L, phi) - (tau_2 - b_2 * phi_dot)

L_1_terms = poly(simplify(L_1), [theta_ddot, phi_ddot]).coeffs()
L_2_terms = poly(simplify(L_2), [theta_ddot, phi_ddot]).coeffs()

mat = Matrix([L_1_terms, L_2_terms])

# th is the initial angle (degrees)
# w is the initial angular velocity (degrees per second)
θ1_initial = -91.0
ω1_initial = 0.0
θ2_initial = 0.0
ω2_initial = 0.0

initial_state = np.radians([θ1_initial, ω1_initial, θ2_initial, ω2_initial])

constants = [(g, G), (l_1, L1), (m_1, M1), (I_1, I1), (l_2, L2), (m_2, M2), (I_2, I2), (r_1, R1), (r_2, R2),
               (b_1, B1), (b_2, B2), (tau_1, T1), (tau_2, T2)]

solution = solve_linear_system(mat.subs(constants), theta_ddot, phi_ddot)


def double_pendulum_deriv(this_state, time_step):
    print(time_step)
    θ1, ω1, θ2, ω2 = this_state

    variables = [(theta, θ1), (theta_dot, ω1), (phi, θ2),(phi_dot, ω2)]

    θ1_dot = ω1
    θ2_dot = ω2

    ω1_dot = float(solution[theta_ddot].subs(variables).doit())
    ω2_dot = float(solution[phi_ddot].subs(variables).doit())

    return np.array([θ1_dot, ω1_dot, θ2_dot, ω2_dot])

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 1, dt)

# integrate your ODE using scipy.integrate.
pos = integrate.odeint(double_pendulum_deriv, initial_state, t)

x1_pos = L1 * np.cos(pos[:, 0])
y1_pos = L1 * np.sin(pos[:, 0])

x2_pos = x1_pos + L2 * np.cos(pos[:, 0] + pos[:, 2])
y2_pos = y1_pos + L2 * np.sin(pos[:, 0] + pos[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
ax.set_aspect('equal', adjustable='box')

line, = ax.plot([], [], 'k-', lw=4, solid_capstyle='round')
time_template = 'time = %.2fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1_pos[i], x2_pos[i]]
    thisy = [0, y1_pos[i], y2_pos[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(pos),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()