def clamped(x):
    import numpy as np
    return 30*(np.clip((x*2)/np.pi,-30,30))

def cube(x):
     import numpy as np
     return 30*(((x*2)/np.pi)**3)

def identity(x):
     import numpy as np
     return 30*((x*2)/np.pi)

def sin(x):
    import numpy as np
    return 30*np.sin((x*2)/np.pi)

def tanh(x):
    import numpy as np
    return 30*np.tanh((x*2)/np.pi)

def get_functions():
    return [("clamped_", clamped),
            ("cube_", cube),
            ("identity_", identity),
            ("tanh_", tanh)]
