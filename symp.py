from sympy import *

x1,x2,y1,y2 = symbols('x1 x2 y1 y2')

x,y,chi,v = symbols('x y chi v')

dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)


x_dot = v  * cos(chi)
y_dot = v  * sin(chi)

print(diff(x_dot, v))
print(diff(x_dot, chi))
print(diff(y_dot, v))
print(diff(y_dot, chi))