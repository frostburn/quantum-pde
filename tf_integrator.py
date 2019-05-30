import numpy as np
import tensorflow as tf

MAX_DT = 0.1

class WaveFunction2D(object):
    def __init__(self, psi, potential, dx=1, dt=MAX_DT, damping_field=None, fixed_point_iterations=3, extra_iterations=0):
        self.set_field(psi)
        self.fixed_point_iterations = fixed_point_iterations

        dt /= dx*dx
        iterations = 1
        while dt > MAX_DT * iterations:
            iterations += 1
        self.iterations = iterations + extra_iterations
        dt /= self.iterations

        width, height = potential.shape
        self.potential = tf.reshape(tf.constant(np.real(potential) * dt * dx * dx, dtype="float64"), [1, width, height, 1])
        self.kernel = tf.reshape(tf.constant(np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]) * dt, dtype="float64"), [3, 3, 1, 1])
        if damping_field is None:
            self.damping_field = None
        else:
            self.damping_field = tf.reshape(tf.constant(np.real(damping_field), dtype="float64"), [1, width, height, 1])

        def integrator(psi_real, psi_imag):
            for _ in range(self.iterations):
                psi_new_real = psi_real
                psi_new_imag = psi_imag
                for _ in range(self.fixed_point_iterations):
                    temp = psi_new_real
                    psi_new_real = psi_real - tf.nn.conv2d(psi_new_imag, self.kernel, 1, 'SAME') + self.potential * psi_new_imag
                    psi_new_imag = psi_imag + tf.nn.conv2d(temp, self.kernel, 1, 'SAME') - self.potential * temp
                psi_real = psi_new_real
                psi_imag = psi_new_imag
            if self.damping_field is None:
                return psi_real, psi_imag
            return psi_real * self.damping_field, psi_imag * self.damping_field

        self.integrator = tf.function(integrator)

    def step(self):
        self.psi_real, self.psi_imag = self.integrator(self.psi_real, self.psi_imag)


    def get_field(self):
        _, width, height, _ = self.psi_real.shape
        return self.psi_real.numpy().reshape([width, height]) + self.psi_imag.numpy().reshape([width, height]) * 1j

    def set_field(self, psi):
        width, height = psi.shape
        self.psi_real = tf.reshape(tf.constant(np.real(psi), dtype="float64"), [1, width, height, 1])
        self.psi_imag = tf.reshape(tf.constant(np.imag(psi), dtype="float64"), [1, width, height, 1])



# from pylab import *

# x = linspace(-5, 5, 100)
# x, y = meshgrid(x, x)

# psi = exp(-x*x - y*y + 5j*x - 1j*y)
# potential = exp(-(x-3) ** 2)
# damping = exp(-(0.15*x)**12 - (0.15*y)**12)

# wave_function = WaveFunction2D(psi, potential, damping_field=damping)

# while True:
#     for _ in range(100):
#         wave_function.step()

#     psi_default = wave_function.get_field()

#     print(abs(psi_default).max())

#     imshow(abs(psi_default))
#     show()

# asdasd

# wave_function = WaveFunction2D(psi, potential, dt=0.1*MAX_DT, damping_field=pow(damping, 0.1))

# for _ in range(500):
#     wave_function.step()

# psi_accurate = wave_function.get_field()

# print(abs(psi_accurate - psi_default).max())


# asdfsdf

# x = linspace(-5, 5, 400)
# x, y = meshgrid(x, x)

# psi = exp(-x*x - y*y + 5j*x - 1j*y)
# potential = exp(-(x-3) ** 2)

# wave_function = WaveFunction2D(psi, potential, dx=0.5)

# for _ in range(500):
#     wave_function.step()

# psi_half = wave_function.get_field()[::2,::2]

# # print(abs(psi_half - psi_accurate).max())
# print(abs(psi_half - psi_default).max())

# imshow(abs(psi_half))
# show()

# # imshow(abs(psi_half - psi_default))
# # show()
