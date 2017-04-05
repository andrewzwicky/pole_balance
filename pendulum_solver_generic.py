from sympy import symbols, init_printing, S, Function, Derivative, diff, simplify, solve, lambdify, nsolve, Matrix, \
    collect, expand, poly, solve_linear_system, cos, sin, latex, Add
import sympy
from sympy.physics.vector import vlatex
from sympy.abc import i, j, k, n
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import matplotlib.animation as animation

init_printing(latex_printer=vlatex)

symbolic = False
N = 2

t = symbols('t')

if symbolic:
    g = symbols('g')
    l = list(symbols('l0:{0}'.format(N)))
    m = list(symbols('m0:{0}'.format(N)))
    r = list(symbols('r0:{0}'.format(N)))
    I = list(symbols('I0:{0}'.format(N)))
    tau = list(symbols('tau0:{0}'.format(N)))
    b = list(symbols('b0:{0}'.format(N)))
else:
    g = S(9.8)
    l = [S(1.0)] * N
    m = [S(1.0)] * N
    r = [a / 2 for a in l]
    I = [(a * (b ** 2)) / 12 for a, b in zip(m, l)]
    tau = [S(2), S(5)]
    b = [S(5.0)] * N

theta = [w(t) for w in symbols('theta0:{0}'.format(N))]
theta_dot = [Derivative(w, t) for w in theta]
theta_ddot = [Derivative(w, t, t) for w in theta]

x = [sum((r[i] if i == n else l[i]) * sympy.cos(sum(theta[j] for j in range(i + 1))) for i in range(n + 1)) for n in
     range(N)]
y = [sum((r[i] if i == n else l[i]) * sympy.sin(sum(theta[j] for j in range(i + 1))) for i in range(n + 1)) for n in
     range(N)]

x_dot = [diff(x[i], t) for i in range(N)]
y_dot = [diff(y[i], t) for i in range(N)]

K = sum([(m[i] * (x_dot[i] ** 2 + y_dot[i] ** 2) + I[i] * (sum([theta_dot[j] for j in range(i + 1)])) ** 2) / 2 for i in
         range(N)])
U = sum([m[i] * g * y[i] for i in range(N)])
L = [simplify(diff((K - U), theta_dot[i], t) - diff((K - U), theta[i]) - (tau[i] - b[i]*theta_dot[i])) for i in range(N)]

solution = solve(L, theta_ddot)

LS = [lambdify((j, i, n, k), solution[theta_ddot[p]].subs([(theta_dot[0], i), (theta[0], j), (theta_dot[1], k), (theta[1], n)])) for p in range(N)]


def double_pendulum_deriv(this_state, time_step):
    this_theta, this_theta_dot, this_phi, this_phi_dot = this_state

    next_theta_dot = this_theta_dot
    next_phi_dot = this_phi_dot

    next_theta_ddot = LS[0](*this_state)
    next_phi_ddot = LS[1](*this_state)

    return np.array([next_theta_dot, next_theta_ddot, next_phi_dot, next_phi_ddot])


if not symbolic:
    dt = 0.05
    t = np.arange(0.0, 10, dt)

    theta_initial = -90.0
    theta_dot_initial = 0.0
    phi_initial = 90.0
    phi_dot_initial = 0.0

    initial_state = np.radians([theta_initial, theta_dot_initial, phi_initial, phi_dot_initial])

    pos = integrate.odeint(double_pendulum_deriv, initial_state, t)

    x1_pos = float(l[0]) * np.cos(pos[:, 0])
    y1_pos = float(l[0]) * np.sin(pos[:, 0])

    x2_pos = x1_pos + float(l[1]) * np.cos(pos[:, 0] + pos[:, 2])
    y2_pos = y1_pos + float(l[1]) * np.sin(pos[:, 0] + pos[:, 2])

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
