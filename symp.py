from sympy import *


inputs = 'm v T D chi vx vy'

# ----------------
outputs = {}
inputs_unpacked = ','.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------

outputs['L_dot'] = sqrt(vx**2 + vy**2)


# ------------------
for oname in outputs:
    print()
    for iname in input_symbs:
        deriv = diff(outputs[oname], iname)
        if deriv != 0:
            st = "\t\tpartials['%s', '%s'] = %s" % (oname, iname, deriv)
            print(st)