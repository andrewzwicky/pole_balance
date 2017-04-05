

```python
from sympy import symbols, init_printing,S,Function,Derivative,diff,simplify,solve,lambdify, nsolve, Matrix, collect, expand, poly, solve_linear_system, cos, sin, latex, Add
import sympy
from sympy.physics.vector import vlatex
from sympy.abc import i,j,k,l,m,n
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import Math, display

init_printing(latex_printer=vlatex)
```


```python
def dis(expr):
    display(Math(vlatex(simplify(expr))))
```


```python
t = symbols('t')
g = symbols('g')

l = list(symbols('l0:2'))
m = list(symbols('m0:2'))
r = list(symbols('r0:2'))
I = list(symbols('I0:2'))
tau = list(symbols('tau0:2'))
b = list(symbols('b0:2'))

theta = list(w(t) for w in symbols('theta0:2'))
theta_dot = [Derivative(w, t) for w in theta]
theta_ddot = [Derivative(w, t, t) for w in theta]

x = [None] * 2
y = [None] * 2
x_dot = [None] * 2
y_dot = [None] * 2

x[0] = r[0] * sympy.cos(theta[0])
y[0] = r[0] * sympy.sin(theta[0])
x[1] = l[1] * sympy.cos(theta[0]) + r[1] * sympy.cos(theta[0] + theta[1])
y[1] = l[1] * sympy.sin(theta[0]) + r[1] * sympy.sin(theta[0] + theta[1])

x_dot[0] = diff(x[0], t)
y_dot[0] = diff(y[0], t)
x_dot[1] = diff(x[1], t)
y_dot[1] = diff(y[1], t)

K = (1/2) * (m[0] * (x_dot[0] ** 2 + y_dot[0] ** 2)
           + m[1] * (x_dot[1] ** 2 + y_dot[1] ** 2)
           + I[0] * (theta_dot[0]               )**2
           + I[1] * (theta_dot[0] + theta_dot[1])**2)
U = (m[0] * g * y[0]) + (m[1] * g * y[1])

L = K - U

L_0 = diff(L, theta_dot[0], t) - diff(L, theta[0]) - (tau[0] + b[0]*theta_dot[0])
L_1 = diff(L, theta_dot[1], t) - diff(L, theta[1]) - (tau[1] + b[1]*theta_dot[1])
```


```python
dis(K)
dis(U)
dis(L_0)
dis(L_1)
```


$$0.5 I_{0} \dot{\theta}_{0}^{2} + 0.5 I_{1} \left(\dot{\theta}_{0} + \dot{\theta}_{1}\right)^{2} + 0.5 m_{0} r_{0}^{2} \dot{\theta}_{0}^{2} + 0.5 m_{1} \left(l_{1}^{2} \dot{\theta}_{0}^{2} + 2 l_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) \dot{\theta}_{0}^{2} + 2 l_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) \dot{\theta}_{0} \dot{\theta}_{1} + r_{1}^{2} \dot{\theta}_{0}^{2} + 2 r_{1}^{2} \dot{\theta}_{0} \dot{\theta}_{1} + r_{1}^{2} \dot{\theta}_{1}^{2}\right)$$



$$g \left(m_{0} r_{0} \operatorname{sin}\left(\theta_{0}\right) + m_{1} \left(l_{1} \operatorname{sin}\left(\theta_{0}\right) + r_{1} \operatorname{sin}\left(\theta_{0} + \theta_{1}\right)\right)\right)$$



$$1.0 I_{0} \ddot{\theta}_{0} + I_{1} \left(\ddot{\theta}_{0} + \ddot{\theta}_{1}\right) - b_{0} \dot{\theta}_{0} + g m_{0} r_{0} \operatorname{cos}\left(\theta_{0}\right) + g m_{1} \left(l_{1} \operatorname{cos}\left(\theta_{0}\right) + r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right)\right) + m_{0} r_{0}^{2} \ddot{\theta}_{0} + m_{1} \left(l_{1}^{2} \ddot{\theta}_{0} - 2 l_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0} \dot{\theta}_{1} - l_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{1}^{2} + 2 l_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) \ddot{\theta}_{0} + l_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) \ddot{\theta}_{1} + r_{1}^{2} \ddot{\theta}_{0} + r_{1}^{2} \ddot{\theta}_{1}\right) - \tau_{0}$$



$$1.0 I_{1} \ddot{\theta}_{0} + 1.0 I_{1} \ddot{\theta}_{1} - 1.0 b_{1} \dot{\theta}_{1} + 1.0 g m_{1} r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right) + 1.0 l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0}^{2} + 1.0 l_{1} m_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) \ddot{\theta}_{0} + 1.0 m_{1} r_{1}^{2} \ddot{\theta}_{0} + 1.0 m_{1} r_{1}^{2} \ddot{\theta}_{1} - 1.0 \tau_{1}$$



```python
solution = solve((L_0, L_1),theta_ddot)
```


```python
dis(solution[theta_ddot[0]])
dis(solution[theta_ddot[1]])
```


$$\frac{1}{I_{0} I_{1} + I_{0} m_{1} r_{1}^{2} + I_{1} l_{1}^{2} m_{1} + I_{1} m_{0} r_{0}^{2} + l_{1}^{2} m_{1}^{2} r_{1}^{2} \operatorname{sin}^{2}\left(\theta_{1}\right) + m_{0} m_{1} r_{0}^{2} r_{1}^{2}} \left(\left(I_{1} + m_{1} r_{1}^{2}\right) \left(b_{0} \dot{\theta}_{0} - g l_{1} m_{1} \operatorname{cos}\left(\theta_{0}\right) - g m_{0} r_{0} \operatorname{cos}\left(\theta_{0}\right) - g m_{1} r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right) + 2.0 l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0} \dot{\theta}_{1} + l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{1}^{2} + \tau_{0}\right) - \left(I_{1} + l_{1} m_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) + m_{1} r_{1}^{2}\right) \left(b_{1} \dot{\theta}_{1} - g m_{1} r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right) - l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0}^{2} + \tau_{1}\right)\right)$$



$$\frac{1}{I_{0} I_{1} + I_{0} m_{1} r_{1}^{2} + I_{1} l_{1}^{2} m_{1} + I_{1} m_{0} r_{0}^{2} + l_{1}^{2} m_{1}^{2} r_{1}^{2} \operatorname{sin}^{2}\left(\theta_{1}\right) + m_{0} m_{1} r_{0}^{2} r_{1}^{2}} \left(- \left(I_{1} + l_{1} m_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) + m_{1} r_{1}^{2}\right) \left(b_{0} \dot{\theta}_{0} - g l_{1} m_{1} \operatorname{cos}\left(\theta_{0}\right) - g m_{0} r_{0} \operatorname{cos}\left(\theta_{0}\right) - g m_{1} r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right) + 2.0 l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0} \dot{\theta}_{1} + l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{1}^{2} + \tau_{0}\right) + \left(b_{1} \dot{\theta}_{1} - g m_{1} r_{1} \operatorname{cos}\left(\theta_{0} + \theta_{1}\right) - l_{1} m_{1} r_{1} \operatorname{sin}\left(\theta_{1}\right) \dot{\theta}_{0}^{2} + \tau_{1}\right) \left(I_{0} + I_{1} + l_{1}^{2} m_{1} + 2.0 l_{1} m_{1} r_{1} \operatorname{cos}\left(\theta_{1}\right) + m_{0} r_{0}^{2} + m_{1} r_{1}^{2}\right)\right)$$



```python

```
