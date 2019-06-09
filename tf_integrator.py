import numpy as np
import tensorflow as tf

MAX_DT = 0.1

class WaveFunction2D(object):
    def __init__(self, psi, potential, dx=1, dt=MAX_DT, damping_field=None, fixed_point_iterations=3, extra_iterations=0, renormalize=False):
        self.set_field(psi)
        self.fixed_point_iterations = fixed_point_iterations

        dt /= dx*dx
        iterations = 1
        while dt > MAX_DT * iterations:
            iterations += 1
        self.iterations = iterations + extra_iterations
        dt /= self.iterations

        width, height = potential.shape
        self.potential_scale = dt * dx * dx
        self.potential = tf.reshape(tf.constant(np.real(potential) * self.potential_scale, dtype="float64"), [1, width, height, 1])
        self.kernel = tf.reshape(tf.constant(np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]) * dt, dtype="float64"), [3, 3, 1, 1])
        if damping_field is None:
            self.damping_field = None
        else:
            self.damping_field = tf.reshape(tf.constant(np.real(damping_field), dtype="float64"), [1, width, height, 1])

        renormalize = True
        self.renormalize = renormalize
        if renormalize:
            self.dx = dx

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
            if self.renormalize:
                total_probability = tf.reduce_sum(psi_real*psi_real + psi_imag*psi_imag) * self.dx**2
                psi_real /= tf.sqrt(total_probability)
                psi_imag /= tf.sqrt(total_probability)
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

    def get_visual(self, screen, hide_phase=False, contrast=4, potential_contrast=1, show_momentum=False):
        _, width, height, _ = self.psi_real.shape
        potential = tf.reshape(self.potential, [width, height])[screen] / (self.potential_scale * 1000) * potential_contrast
        if show_momentum:
            psi = tf.cast(tf.reshape(self.psi_real, [width, height]), dtype=tf.complex128)
            psi += tf.cast(tf.reshape(self.psi_imag, [width, height]), dtype=tf.complex128) * 1j
            psi = tf.signal.fft2d(psi)
            w = width // 2
            h = height // 2
            psi = tf.concat([psi[w:], psi[:w]], axis=0)
            psi = tf.concat([psi[:, h:], psi[:, :h]], axis=1)
            psi = psi[screen]
        else:
            psi = tf.cast(tf.reshape(self.psi_real, [width, height])[screen], dtype=tf.complex128)
            psi += tf.cast(tf.reshape(self.psi_imag, [width, height])[screen], dtype=tf.complex128) * 1j
        amplitude = tf.tanh(tf.math.abs(psi) * contrast)
        phase = tf.math.angle(psi)
        band1 = (phase / np.pi) ** 10
        band2 = ((1.05 + phase / np.pi) % 2 - 1) ** 16
        band3 = ((0.95 + phase / np.pi) % 2 - 1) ** 16
        rgb = tf.stack([band2, band3, band1])
        if hide_phase:
            rgb *= 0
        rgb *= amplitude ** 0.8
        rgb += tf.stack([amplitude, amplitude, 0.7*amplitude])
        if not show_momentum:
            rgb += tf.stack([potential*0.2, potential*0.7, potential*0.4])
        return rgb.numpy()
