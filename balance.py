"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c
# http://www.physics.usyd.edu.au/~wheat/dpend_html/

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

s = np.sin
c = np.cos

G = 9.8  # acceleration due to gravity, in m/s^2
LS = 1.0  # length of pendulum 1 in m
LP = 1.0  # length of pendulum 2 in m
MS = 1.0  # mass of pendulum 1 in kg
MP = 1.0  # mass of pendulum 2 in kg
RS = LS/2
RP = LP/2
IP = (MP*LP*LP)/3
IS = (MS*LS*LS)/3
TS = MS*G*LS*c(np.radians(285))
TP = 0

def derivs(this_state, this_t):

    ths, ws, thp, wp = this_state

    ths_dot = ws

    delta_th = thp - ths
    den1 = (MS + MP) * LS - MP * LS * cos(delta_th) * cos(delta_th)
    ws_dot = (MP * LS * ws * ws * sin(delta_th) * cos(delta_th) +
               MP * G * sin(thp) * cos(delta_th) +
               MP * LP * wp * wp * sin(delta_th) -
               (MS + MP) * G * sin(ths)) / den1

    thp_dot = wp

    den2 = (LP / LS) * den1
    wp_dot = (-MP * LP * wp * wp * sin(delta_th) * cos(delta_th) +
               (MS + MP) * G * sin(ths) * cos(delta_th) -
               (MS + MP) * LS * ws * ws * sin(delta_th) -
               (MS + MP) * G * sin(thp)) / den2

    return np.array([ths_dot, ws_dot, thp_dot, wp_dot])

def single_with_torque_deriv(this_state, this_t, torque, damping):
    ths, ws = this_state

    ths_dot = ws
    #ws_dot = (-MS*G*LS*c(ths))/(2*IS)
    ws_dot = (torque - .1*ws - (MS * G * LS * c(ths) / 2)) / IS

    return np.array([ths_dot, ws_dot])

def robot_derivs(this_state, this_t):
    ths, ws, thp, wp = this_state

    a = IP + IS + MS*RS*RS + MP*(LS*LS+RP*RP)
    b = MP*LS*RP
    d = IP+MP*RP*RP

    C = np.matrix([[-b*s(thp)*wp, -b*s(thp)*(ws+wp)],
                   [b*s(thp)*ws , 0]])

    D = np.matrix([[a+2*b*c(thp), d+b*c(thp)],
                   [d+b*c(thp)  , d]])

    U = np.matrix([[MS*G*RS*s(ths)],
                   [MP*G*(LS*s(ths)+RP*s(ths+thp))]])

    T = np.matrix([[TS],[TP]])
    W = np.matrix([[ws],[wp]])

    ws_dot, wp_dot = np.linalg.solve(D, T-U-np.matmul(C,W))

    ths_dot = ws
    thp_dot = wp

    return ths_dot, ws_dot, thp_dot, wp_dot

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
ths = 285.0
ws = 0.0
thp = 90.0
wp = 0.0

# initial state
state = np.radians([ths, ws, thp, wp])
single_state = np.radians([ths, ws])

# integrate your ODE using scipy.integrate.
pos = integrate.odeint(single_with_torque_deriv, single_state, t, args=(TS,))

xs = LS * cos(pos[:, 0])
ys = LS * sin(pos[:, 0])

#xp = LP * sin(pos[:, 2]) + xs
#yp = -LP * cos(pos[:, 2]) + ys

xp = [0]*len(pos)
yp = [0]*len(pos)


fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
plt.gca().set_aspect('equal', adjustable='box')

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, xs[i]]
    thisy = [0, ys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(pos)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()