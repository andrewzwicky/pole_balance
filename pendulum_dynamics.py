from sympy import symbols, init_printing, S, Derivative, diff, simplify, solve, lambdify, cos, sin
from sympy.physics.vector import vlatex
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
from matplotlib import animation, rc
from itertools import chain

rc('animation', html='html5')

init_printing(latex_printer=vlatex, latex_mode='equation')


def generate_double_pendulum_odes():
    """
    :return:
     List of ODE describing system (Number = DOF of system)
     List of plotting position functions (Number = DOF of system)
    """
    t = symbols('t')
    g = symbols('g')
    l = symbols('l0:2')
    m = symbols('m0:2')
    r = symbols('r0:2')
    i = symbols('I0:2')
    tau = symbols('tau0:2')
    b = symbols('b0:2')

    g_val = S(9.8)
    l_val = [S(1.0), S(1.0)]
    m_val = [S(1.0), S(1.0)]
    r_val = [temp_l / 2 for temp_l in l_val]
    i_val = [(temp_m * temp_l ** 2) / 12 for temp_m, temp_l in zip(m_val, l_val)]
    tau_val = [S(0.0), S(0.0)]
    b_val = [S(0.0), S(0.0)]

    theta = [w(t) for w in symbols('theta0:2')]
    theta_dot = [Derivative(w, t) for w in theta]
    theta_ddot = [Derivative(w, t, t) for w in theta]

    x = [None] * 2
    y = [None] * 2
    x_dot = [None] * 2
    y_dot = [None] * 2

    x[0] = r[0] * cos(theta[0])
    y[0] = r[0] * sin(theta[0])
    x[1] = l[1] * cos(theta[0]) + r[1] * cos(theta[0] + theta[1])
    y[1] = l[1] * sin(theta[0]) + r[1] * sin(theta[0] + theta[1])

    x_dot[0] = diff(x[0], t)
    y_dot[0] = diff(y[0], t)
    x_dot[1] = diff(x[1], t)
    y_dot[1] = diff(y[1], t)

    kinetic = (m[0] * (x_dot[0] ** 2 + y_dot[0] ** 2)
               + m[1] * (x_dot[1] ** 2 + y_dot[1] ** 2)
               + i[0] * (theta_dot[0]) ** 2
               + i[1] * (theta_dot[0] + theta_dot[1]) ** 2) / 2

    potential = (m[0] * g * y[0]) + (m[1] * g * y[1])

    lagrange = kinetic - potential

    lagrangian = [None] * 2
    lagrangian[0] = diff(lagrange, theta_dot[0], t) - diff(lagrange, theta[0])
    lagrangian[1] = diff(lagrange, theta_dot[1], t) - diff(lagrange, theta[1])

    solution = solve(lagrangian, theta_ddot)

    values = [(g, g_val),
              (l[0], l_val[0]),
              (l[1], l_val[1]),
              (m[0], m_val[0]),
              (m[1], m_val[1]),
              (r[0], r_val[0]),
              (r[1], r_val[1]),
              (i[0], i_val[0]),
              (i[1], i_val[1]),
              (tau[0], tau_val[0]),
              (tau[1], tau_val[1]),
              (b[0], b_val[0]),
              (b[1], b_val[1])]

    temp_vars = symbols('z0:4')

    inputs = list(zip((theta_dot[0], theta[0], theta_dot[1], theta[1]), temp_vars))

    ode_equations = [None] * 2
    ode_equations[0] = lambdify(temp_vars, simplify(solution[theta_ddot[0]]).subs(values).subs(inputs))
    ode_equations[1] = lambdify(temp_vars, simplify(solution[theta_ddot[1]]).subs(values).subs(inputs))

    def double_pendulum_position(pos):
        result = []

        for theta0, _, theta1, _ in pos:
            x1_pos = float(l_val[0]) * np.cos(theta0)
            y1_pos = float(l_val[0]) * np.sin(theta0)

            x2_pos = x1_pos + float(l_val[1]) * np.cos(theta0 + theta1)
            y2_pos = y1_pos + float(l_val[1]) * np.sin(theta0 + theta1)

            result.append(((0, x1_pos, x2_pos), (0, y1_pos, y2_pos)))

        return result

    return ode_equations, double_pendulum_position


def generic_deriv_handler(this_state, _, deriv_functions):
    result = [(this_state[(i * 2) + 1], float(func(*this_state))) for i, func in enumerate(deriv_functions)]

    flattened = chain.from_iterable(result)
    float_flattened = list(map(float,flattened))

    return np.array(float_flattened)


def animate_system(time, time_step, initial_conditions, derivation_functions, position_function):

    pos = integrate.odeint(generic_deriv_handler, np.radians(initial_conditions), np.arange(0.0, time, time_step),
                           args=(derivation_functions,))

    plot_positions = position_function(pos)

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
        # TODO position func should be generator
        # TODO dependance on DOF
        thisx, thisy = plot_positions[i]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template.format(time_step))
        return line, time_text

    return animation.FuncAnimation(fig, animate, frames=len(pos), interval=25, blit=True, init_func=init)


ani = animate_system(5, 0.05, [0, 0, 0, 0], *generate_double_pendulum_odes())
plt.show()
