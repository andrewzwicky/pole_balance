import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import sympy
from sympy import symbols, diff, Function, Derivative, poly, simplify, Matrix, init_printing, solve_linear_system, latex, S, nsolve, relational
from sympy.physics.vector import vlatex
from IPython.display import display, Math, Latex
from sympy.abc import i,j,k,l,m,n


t = symbols('t')

g = S(9.8)  # acceleration due to gravity, in m/s^2
l_1 = S(1.0)  # length of pendulum 1 in m
m_1 = S(1.0)  # mass of pendulum 1 in kg
l_2 = S(1.0)  # length of pendulum 2 in m
m_2 = S(1.0)  # mass of pendulum 2 in kg
theta = Function('theta')(t)
phi = Function('phi')(t)
theta_dot = Derivative(theta, t)
phi_dot = Derivative(phi, t)
theta_ddot = Derivative(theta, t, 2)
phi_ddot = Derivative(phi, t, 2)

x1 = l_1 * sympy.cos(theta)
y1 = l_1 * sympy.sin(theta)
x2 = x1 + l_2 * sympy.cos(theta+phi)
y2 = y1 + l_2 * sympy.sin(theta+phi)

x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

K = (1/2)*(m_1 * (x1_dot ** 2 + y1_dot ** 2) + m_2 * (x2_dot ** 2 + y2_dot ** 2))
U = (m_1 * g * y1) + (m_2 * g * y2)
L = K - U

L_1 = diff(L, theta_dot, t) - diff(L, theta)
L_2 = diff(L, phi_dot, t) - diff(L, phi)


def double_pendulum_deriv(this_state, time_step):
    print(time_step)
    this_theta, this_theta_dot, this_phi, this_phi_dot = this_state

    variables = [(theta_ddot,i), (theta_dot, this_theta_dot), (theta, this_theta), (phi_ddot, l), (phi_dot, this_phi_dot), (phi, this_phi)]

    next_theta_dot = this_theta_dot
    next_phi_dot = this_phi_dot

    eqs = (L_1.subs(variables), L_2.subs(variables))

    solution = nsolve(eqs, (i, l), (0, 0))

    next_theta_ddot = float(solution[0].doit())
    next_phi_ddot = float(solution[1].doit())

    return np.array([next_theta_dot, next_theta_ddot, next_phi_dot, next_phi_ddot])

dt = 0.05
t = np.arange(0.0, 10, dt)

theta_initial = 10.0
theta_dot_initial = 0.0
phi_initial = 0.0
phi_dot_initial = 0.0

initial_state = np.radians([theta_initial, theta_dot_initial, phi_initial, phi_dot_initial])

pos = integrate.odeint(double_pendulum_deriv, initial_state, t)

x1_pos = float(l_1) * np.cos(pos[:, 0])
y1_pos = float(l_1) * np.sin(pos[:, 0])

x2_pos = x1_pos + float(l_2) * np.cos(pos[:, 0] + pos[:, 2])
y2_pos = y1_pos + float(l_2) * np.sin(pos[:, 0] + pos[:, 2])

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