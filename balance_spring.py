from sympy import symbols, init_printing, S, Function, Derivative, diff, simplify, solve, lambdify, nsolve, Matrix, \
    collect, expand, poly, solve_linear_system, cos, sin, latex, Add
import sympy
from sympy.physics.vector import vlatex
from sympy.abc import i, j, n, p, o
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import matplotlib.animation as animation

init_printing(latex_printer=vlatex)

t = symbols('t')

g = S(9.8)
k = S(1.0)
l = [S(1.0)] * 2
m = [S(1.0)] * 3
r = [temp_l/2 for temp_l in l]
I = [(temp_m * temp_l**2)/12 for temp_m,temp_l in zip(m,l)]

theta = list(w(t) for w in symbols('theta0:2'))
theta_dot = [Derivative(w, t) for w in theta]
theta_ddot = [Derivative(w, t, t) for w in theta]

d = Function('d')(t)
d_dot = Derivative(d, t)
d_ddot = Derivative(d, t, t)

x = [None] * 3
y = [None] * 3
x_dot = [None] * 3
y_dot = [None] * 3

x[0] = r[0] * cos(theta[0]) + d
y[0] = r[0] * sin(theta[0])
x[1] = l[1] * cos(theta[0]) + r[1] * cos(theta[0] + theta[1]) + d
y[1] = l[1] * sin(theta[0]) + r[1] * sin(theta[0] + theta[1])
x[2] = d

x_dot[0] = diff(x[0], t)
y_dot[0] = diff(y[0], t)
x_dot[1] = diff(x[1], t)
y_dot[1] = diff(y[1], t)
x_dot[2] = diff(x[2], t)

kinetic = (m[0] * (x_dot[0] ** 2 + y_dot[0] ** 2)
           + m[1] * (x_dot[1] ** 2 + y_dot[1] ** 2)
           + I[0] * (theta_dot[0]               )**2
           + I[1] * (theta_dot[0] + theta_dot[1])**2
           + m[2] * (x_dot[2] ** 2)) / 2

potential = (m[0] * g * y[0]) + (m[1] * g * y[1]) + (k*d**2)/2
lagrange = kinetic - potential

L = [None] * 3
L[0] = diff(lagrange, theta_dot[0], t) - diff(lagrange, theta[0])
L[1] = diff(lagrange, theta_dot[1], t) - diff(lagrange, theta[1])
L[2] = diff(lagrange, d_dot, t) - diff(lagrange, d)

solution = solve(L, [theta_ddot[0], theta_ddot[1], d_ddot])

inputs = [(theta_dot[0], i), (theta[0], j), (theta_dot[1], k), (theta[1], n), (d_dot, p), (d, o)]

LS = []

LS[0] = lambdify((j, i, n, k, o, p), simplify(solution[theta_ddot[0]]).subs(inputs))
LS[1] = lambdify((j, i, n, k, o, p), simplify(solution[theta_ddot[1]]).subs(inputs))
LS[2] = lambdify((j, i, n, k, o, p), simplify(solution[d_ddot]).subs(inputs))


def double_pendulum_deriv(this_state, time_step):
    this_theta, this_theta_dot, this_phi, this_phi_dot, this_d, this_d_dot = this_state

    next_theta_dot = this_theta_dot
    next_phi_dot = this_phi_dot
    next_d_dot = this_d_dot

    next_theta_ddot = float(LS[0](*this_state))
    next_phi_ddot = float(LS[1](*this_state))
    next_d_ddot = float(LS[2](*this_state))

    return np.array([next_theta_dot, next_theta_ddot, next_phi_dot, next_phi_ddot, next_d_dot, next_d_ddot])


dt = 0.05
t = np.arange(0.0, 10, dt)

theta_initial = -90.0
theta_dot_initial = 0.0
phi_initial = 90.0
phi_dot_initial = 0.0
d_initial = 0.0
d_dot_initial = 0.0

initial_state = np.radians([theta_initial, theta_dot_initial, phi_initial, phi_dot_initial,d_initial,d_dot_initial])

pos = integrate.odeint(double_pendulum_deriv, initial_state, t)

x0_pos = float(pos[:, 4])
y0_pos = float(pos[:, 4])

x1_pos = float(l[0]) * np.cos(pos[:, 0] + pos[:, 4])
y1_pos = float(l[0]) * np.sin(pos[:, 0] + pos[:, 4])

x2_pos = x1_pos + float(l[1]) * np.cos(pos[:, 0] + pos[:, 2] + pos[:, 4])
y2_pos = y1_pos + float(l[1]) * np.sin(pos[:, 0] + pos[:, 2] + pos[:, 4])

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
    thisx = [x0_pos[i], x1_pos[i], x2_pos[i]]
    thisy = [y0_pos[i], y1_pos[i], y2_pos[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(pos),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()