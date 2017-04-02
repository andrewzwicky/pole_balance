from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L = 1.0  # length of pendulum 1 in m
M = 1.0  # mass of pendulum 1 in kg
I = (M * L * L) / 3  # moment of inertia in kg*m^2 , fixed at end


def single_pendulum_deriv(this_state, time_step, torque_i, damping_i):
    θ, ω = this_state

    θ_dot = ω
    ω_dot = (torque_i - damping_i * ω - (M * G * L * cos(θ) / 2)) / I

    return np.array([θ_dot, ω_dot])


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th is the initial angle (degrees)
# w is the initial angular velocity (degrees per second)
θ_initial = 0.0
ω_initial = 0.0

initial_state = np.radians([θ_initial, ω_initial])

torque = G/4  # torque in Nm
damping = 5

# integrate your ODE using scipy.integrate.
pos = integrate.odeint(single_pendulum_deriv, initial_state, t, args=(torque, damping))

x_pos = L * cos(pos[:, 0])
y_pos = L * sin(pos[:, 0])

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
    thisx = [0, x_pos[i]]
    thisy = [0, y_pos[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, frames=len(pos),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()
