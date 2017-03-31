from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
M1 = 1.0  # mass of pendulum 1 in kg
I1 = (M1 * L1 * L1) / 3  # moment of inertia of pendulum 1 , fixed at end
L2 = 1.0  # length of pendulum 1 in m
M2 = 1.0  # mass of pendulum 1 in kg
I2 = (M2 * L2 * L2) / 3  # moment of inertia of pendulum 1 , fixed at end


def double_pendulum_deriv(this_state, time_step, torques, dampings):
    θ1, ω1, θ2, ω2 = this_state


    return np.array([θ1_dot, ω1_dot, θ2_dot, ω2_dot])


# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th is the initial angle (degrees)
# w is the initial angular velocity (degrees per second)
θ1_initial = 270.0
ω1_initial = 0.0
θ2_initial = 0.0
ω2_initial = 0.0

initial_state = np.radians([θ1_initial, ω1_initial, θ2_initial, ω2_initial])

torque = 0
damping = 0

# integrate your ODE using scipy.integrate.
pos = integrate.odeint(double_pendulum_deriv, initial_state, t, args=(torque, damping))

x1_pos = L1 * cos(pos[:, 0])
y1_pos = L1 * sin(pos[:, 0])

x2_pos = L1 * cos(pos[:, 0])
y2_pos = L1 * sin(pos[:, 0])

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
