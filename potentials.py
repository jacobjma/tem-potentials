import csv
import numbers

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
from scipy.special import erfc


def find_cutoff(func, min_value, xmin=1e-12, xmax=1000):
    return brentq(lambda x: func(x) - min_value, xmin, xmax)


def log_grid(start, stop, num):
    dt = np.log(stop / start) / (num - 1)
    return start * np.exp(dt * np.linspace(0, num - 1, num))


def potential_spline_coeff(r_min, v_min, n_nodes, v, dvdr):
    r_cut = find_cutoff(v, v_min)

    r = log_grid(r_min, r_cut, n_nodes)

    v = v(r) - v(r_cut) - (r - r_cut) * dvdr(r_cut)

    bc_left, bc_right = [(1, dvdr(r_min))], [(1, 0.)]

    return make_interp_spline(r, v, bc_type=(bc_left, bc_right))


def scattering_spline_coeff(f_min, n_nodes, f, dfdg):
    g_cut = find_cutoff(f, f_min)

    g = np.linspace(0, g_cut, n_nodes)

    f = f(g) - f(g_cut) - (g - g_cut) * dfdg(g_cut)

    bc_left, bc_right = [(1, 0.)], [(1, 0.)]

    return make_interp_spline(g, f, bc_type=(bc_left, bc_right))


def fourier_translation(x, y, z, gx, gy, gz):
    return np.exp(-2 * np.pi * 1j * (gx * x + gy * y + gz * z))


def slice_potential(v, n, Lz):
    Nz = v.shape[2]

    if Nz % n != 0:
        raise RuntimeError('v.shape[2] is not divisible by n'.format(Nz, n))

    v_slices = np.zeros(v.shape[:2] + (n,))

    dz = Lz / Nz
    nz = Nz // n
    for i in range(n):
        v_slices[:, :, i] = np.trapz(v[:, :, i * nz:(i + 1) * nz + 1], dx=dz, axis=2)

    return v_slices, dz


class ParameterizedPotential(object):

    def __init__(self, atoms, grid, parameters, units='atomic'):

        if atoms is None:
            self._atoms = None
        else:
            self._atoms = atoms
            cell = atoms.get_cell()
            if np.any(np.abs(cell[~np.eye(cell.shape[0], dtype=bool)]) > 1e-12):
                raise RuntimeError()

        if grid is None:
            self._grid = None
        elif isinstance(grid, numbers.Integral):
            self._grid = (grid,) * 3
        else:
            self._grid = grid

        if parameters is None:
            self._parameters = None
        elif isinstance(parameters, str):
            self._parameters = self.load(parameters)
        else:
            self._parameters = parameters

        self.units = units

    @property
    def atoms(self):
        return self._atoms

    @property
    def grid(self):
        return self._grid

    @property
    def box(self):
        return np.diag(self._atoms.get_cell())

    def kappa(self):
        if self.units.lower() == 'si':
            # 2 * 8.854187817e-12 / (5.2917721067e-1 * 1.60217662e-19)
            return 208865737.13072053
        elif self.units.lower() == 'atomic':
            # 2 * 7.957747155e-2 / (5.2917721067e-1 * 1)
            return 1  # 0.3007592539718241
        else:
            raise ValueError()

    def voxel_size(self):
        Lx, Ly, Lz = self.box
        Nx, Ny, Nz = self.grid

        return Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)

    def recip_space_coordinates(self):

        Nx, Ny, Nz = self.grid
        dx, dy, dz = self.voxel_size()

        gx, gy, gz = np.fft.fftfreq(Nx, dx), np.fft.fftfreq(Ny, dy), np.fft.fftfreq(Nz, dz)

        g2 = (gx ** 2)[:, None, None] + (gy ** 2)[None, :, None] + (gz ** 2)[None, None, :]

        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')

        return gx, gy, gz, g2

    def real_space_coordinates(self):

        x = np.linspace(0, self.box[0], self.grid[0])
        y = np.linspace(0, self.box[1], self.grid[1])
        z = np.linspace(0, self.box[2], self.grid[2])

        r2 = (x ** 2)[:, None, None] + (y ** 2)[None, :, None] + (z ** 2)[None, None, :]

        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        return x, y, z, r2

    def scattering_splines(self, f_min, n_nodes):

        splines = {}
        for number in np.unique(self.atoms.get_atomic_numbers()):
            f, dfdg = self.scattering_factor(number)
            splines[number] = scattering_spline_coeff(f_min, n_nodes, f, dfdg)

        return splines

    def calc_scattering(self, f_min=1e-2, n_nodes=50):

        splines_lookup = self.scattering_splines(f_min, n_nodes)

        gx, gy, gz, g2 = self.recip_space_coordinates()

        f = np.zeros(self.grid, dtype='complex')

        g = np.sqrt(g2)

        for atom in self.atoms:
            splines = splines_lookup[atom.number]

            inside = g < splines.t.max()

            f[inside] += splines(g[inside]) * fourier_translation(*atom.position, gx[inside], gy[inside], gz[inside])

        return f

    def calc_from_scattering(self, f_min=1e-2, n_nodes=50):
        f = self.calc_scattering(f_min, n_nodes)
        return np.fft.ifftn(f).real / np.prod(self.voxel_size()) / self.kappa()

    def potential_splines(self, r_min, v_min, n_nodes):

        if r_min is None:
            r_min = min(self.voxel_size())

        splines = {}
        for number in np.unique(self.atoms.get_atomic_numbers()):
            v, dvdr = self.analytic_potential(number)

            splines[number] = potential_spline_coeff(r_min, v_min, n_nodes, v, dvdr)

        return splines

    def repeated_positions(self, max_cutoff):

        if np.any(max_cutoff > self.box):
            raise RuntimeError()

        positions = self.atoms.get_positions()
        numbers = self.atoms.get_atomic_numbers()

        lattice_vectors = np.zeros((3, 3))
        np.fill_diagonal(lattice_vectors, self.box)

        for i, lattice_vector in enumerate(lattice_vectors):
            left_indices = np.where(positions[:, i] < max_cutoff)[0]
            left_positions = positions[left_indices] + lattice_vector

            right_indices = np.where((self.box[i] - positions[:, i]) < max_cutoff)[0]
            right_positions = positions[right_indices] - lattice_vector

            positions = np.vstack((positions, left_positions, right_positions))
            numbers = np.hstack((numbers, numbers[left_indices], numbers[right_indices]))

        return positions, numbers

    def calc_from_splines(self, r_min=None, v_min=1e-3, n_nodes=50):

        splines_lookup = self.potential_splines(r_min, v_min, n_nodes)

        max_cutoff = max([splines.t.max() for _, splines in splines_lookup.items()])

        x, y, z, _ = self.real_space_coordinates()

        V = np.zeros(self.grid, dtype='float')
        for position, number in zip(*self.repeated_positions(max_cutoff)):
            xi, yi, zi = position

            splines = splines_lookup[number]

            r2 = (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2

            inside = r2 < splines.t.max() ** 2

            r = np.sqrt(r2[inside])

            V[inside] += splines(r)

        return V / self.kappa()

    def load(self, filename):

        parameters = {}
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            keys = next(reader)[1:]
            for _, row in enumerate(reader):
                values = list(map(float, row[1:]))
                parameters[int(row[0])] = dict(zip(keys, values))

        return parameters


class Lobato(ParameterizedPotential):

    def __init__(self, atoms=None, grid=None, parameters='data/lobato.txt', units='atomic'):
        super(Lobato, self).__init__(atoms, grid, parameters, units)

    def analytic_potential(self, element):
        parameters = self._parameters[element]

        a = [np.pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
             zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
        b = [2 * np.pi / np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

        v = lambda r: (a[0] * (2. / (b[0] * r) + 1) * np.exp(-b[0] * r) +
                       a[1] * (2. / (b[1] * r) + 1) * np.exp(-b[1] * r) +
                       a[2] * (2. / (b[2] * r) + 1) * np.exp(-b[2] * r) +
                       a[3] * (2. / (b[3] * r) + 1) * np.exp(-b[3] * r) +
                       a[4] * (2. / (b[4] * r) + 1) * np.exp(-b[4] * r))

        dvdr = lambda r: - (a[0] * (2 / (b[0] * r ** 2) + 2 / r + b[0]) * np.exp(-b[0] * r) +
                            a[1] * (2 / (b[1] * r ** 2) + 2 / r + b[1]) * np.exp(-b[1] * r) +
                            a[2] * (2 / (b[2] * r ** 2) + 2 / r + b[2]) * np.exp(-b[2] * r) +
                            a[3] * (2 / (b[3] * r ** 2) + 2 / r + b[3]) * np.exp(-b[3] * r) +
                            a[4] * (2 / (b[4] * r ** 2) + 2 / r + b[4]) * np.exp(-b[4] * r))

        return v, dvdr

    def scattering_factor(self, element):
        parameters = self._parameters[element]

        a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')]
        b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

        f = lambda g: (a[0] * (2 + b[0] * g ** 2) / (1 + b[0] * g ** 2) ** 2 +
                       a[1] * (2 + b[1] * g ** 2) / (1 + b[1] * g ** 2) ** 2 +
                       a[2] * (2 + b[2] * g ** 2) / (1 + b[2] * g ** 2) ** 2 +
                       a[3] * (2 + b[3] * g ** 2) / (1 + b[3] * g ** 2) ** 2 +
                       a[4] * (2 + b[4] * g ** 2) / (1 + b[4] * g ** 2) ** 2)

        dfdg = lambda g: - ((2 * a[0] * b[0] * g * (3 + b[0] * g ** 2)) / (1 + b[0] * g ** 2) ** 3 +
                            (2 * a[1] * b[1] * g * (3 + b[1] * g ** 2)) / (1 + b[1] * g ** 2) ** 3 +
                            (2 * a[2] * b[2] * g * (3 + b[2] * g ** 2)) / (1 + b[2] * g ** 2) ** 3 +
                            (2 * a[3] * b[3] * g * (3 + b[3] * g ** 2)) / (1 + b[3] * g ** 2) ** 3 +
                            (2 * a[4] * b[4] * g * (3 + b[4] * g ** 2)) / (1 + b[4] * g ** 2) ** 3)

        return f, dfdg


class Kirkland(ParameterizedPotential):

    def __init__(self, atoms=None, grid=None, parameters='data/kirkland.txt', units='atomic'):
        super(Kirkland, self).__init__(atoms, grid, parameters, units=units)

    def analytic_potential(self, element):
        parameters = self._parameters[element]

        a = [np.pi * parameters[key] for key in ('a1', 'a2', 'a3')]
        b = [2 * np.pi * np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3')]
        c = [np.pi ** (3 / 2.) * parameters[key_c] / parameters[key_d] for key_c, key_d in
             zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))]
        d = [np.pi ** 2 / parameters[key] for key in ('d1', 'd2', 'd3')]

        v = lambda r: (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2) +
                       a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2) +
                       a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2))

        dvdr = lambda r: (- a[0] * (1 / r + b[0]) * np.exp(-b[0] * r) / r - 2 * c[0] * d[0] * r * np.exp(-d[0] * r ** 2)
                          - a[1] * (1 / r + b[1]) * np.exp(-b[1] * r) / r - 2 * c[1] * d[1] * r * np.exp(-d[1] * r ** 2)
                          - a[2] * (1 / r + b[2]) * np.exp(-b[2] * r) / r - 2 * c[2] * d[2] * r * np.exp(-d[2] * r ** 2)
                          )

        return v, dvdr

    def scattering_factor(self, element):
        parameters = self._parameters[element]

        a = [parameters[key] for key in ('a1', 'a2', 'a3')]
        b = [parameters[key] for key in ('b1', 'b2', 'b3')]
        c = [parameters[key] for key in ('c1', 'c2', 'c3')]
        d = [parameters[key] for key in ('d1', 'd2', 'd3')]

        f = lambda g: (a[0] / (b[0] + g ** 2) + c[0] * np.exp(-d[0] * g ** 2) +
                       a[1] / (b[1] + g ** 2) + c[1] * np.exp(-d[1] * g ** 2) +
                       a[2] / (b[2] + g ** 2) + c[2] * np.exp(-d[2] * g ** 2))

        dfdg = lambda g: (- 2 * a[0] * g / (b[0] + g ** 2) ** 2 - 2 * c[0] * d[0] * g * np.exp(-d[0] * g ** 2)
                          - 2 * a[1] * g / (b[1] + g ** 2) ** 2 - 2 * c[1] * d[1] * g * np.exp(-d[1] * g ** 2)
                          - 2 * a[2] * g / (b[2] + g ** 2) ** 2 - 2 * c[2] * d[2] * g * np.exp(-d[2] * g ** 2))

        return f, dfdg


class Peng(ParameterizedPotential):

    def __init__(self, atoms=None, grid=None, parameters='data/peng.txt', units='atomic'):
        super(Peng, self).__init__(atoms, grid, parameters, units=units)

    def analytic_potential(self, element):
        parameters = self._parameters[element]

        a = [parameters[key_a] * np.pi ** (3 / 2.) / parameters[key_b] ** (3 / 2.) for key_a, key_b in
             zip(('a1', 'a2', 'a3', 'a4'), ('b1', 'b2', 'b3', 'b4'))]
        b = [np.pi ** 2 / parameters[key] for key in ('b1', 'b2', 'b3', 'b4')]

        v = lambda r: (a[0] * np.exp(-b[0] * r ** 2) +
                       a[1] * np.exp(-b[1] * r ** 2) +
                       a[2] * np.exp(-b[2] * r ** 2) +
                       a[3] * np.exp(-b[3] * r ** 2))

        dvdr = lambda r: (- 2 * a[0] * b[0] * r * np.exp(-b[0] * r ** 2)
                          - 2 * a[1] * b[1] * r * np.exp(-b[1] * r ** 2)
                          - 2 * a[2] * b[2] * r * np.exp(-b[2] * r ** 2)
                          - 2 * a[3] * b[3] * r * np.exp(-b[3] * r ** 2))

        return v, dvdr

    def scattering_factor(self, element):
        parameters = self._parameters[element]

        a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4')]
        b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4')]

        print(parameters)

        f = lambda g: (a[0] * np.exp(-b[0] * g ** 2) +
                       a[1] * np.exp(-b[1] * g ** 2) +
                       a[2] * np.exp(-b[2] * g ** 2) +
                       a[3] * np.exp(-b[3] * g ** 2))

        dfdg = lambda g: (- 2 * a[0] * b[0] * np.exp(-b[0] * g ** 2)
                          - 2 * a[1] * b[1] * np.exp(-b[1] * g ** 2)
                          - 2 * a[2] * b[2] * np.exp(-b[2] * g ** 2)
                          - 2 * a[3] * b[3] * np.exp(-b[3] * g ** 2))

        return f, dfdg


class Weickenmeier(ParameterizedPotential):

    def __init__(self, atoms=None, grid=None, parameters='data/weickenmeier.txt', units='atomic'):
        super(Weickenmeier, self).__init__(atoms, grid, parameters, units=units)

    def analytic_potential(self, element):
        parameters = self._parameters[element]

        a = 3 * [np.pi * 0.02395 * element / (3 * (1 + parameters['V']))]
        a = a + 3 * [parameters['V'] * a[0]]
        b = [np.pi / np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

        v = lambda r: (a[0] * erfc(b[0] * r) / r +
                       a[1] * erfc(b[1] * r) / r +
                       a[2] * erfc(b[2] * r) / r +
                       a[3] * erfc(b[3] * r) / r +
                       a[4] * erfc(b[4] * r) / r +
                       a[5] * erfc(b[5] * r) / r)

        dvdr = lambda r: (- a[0] * (erfc(b[0] * r) / r ** 2 + 2 * b[0] / np.sqrt(np.pi) * np.exp(-b[0] * r ** 2) / r)
                          - a[1] * (erfc(b[1] * r) / r ** 2 + 2 * b[1] / np.sqrt(np.pi) * np.exp(-b[1] * r ** 2) / r)
                          - a[2] * (erfc(b[2] * r) / r ** 2 + 2 * b[2] / np.sqrt(np.pi) * np.exp(-b[2] * r ** 2) / r)
                          - a[3] * (erfc(b[3] * r) / r ** 2 + 2 * b[3] / np.sqrt(np.pi) * np.exp(-b[3] * r ** 2) / r)
                          - a[4] * (erfc(b[4] * r) / r ** 2 + 2 * b[4] / np.sqrt(np.pi) * np.exp(-b[4] * r ** 2) / r)
                          - a[5] * (erfc(b[5] * r) / r ** 2 + 2 * b[5] / np.sqrt(np.pi) * np.exp(-b[5] * r ** 2) / r))

        return v, dvdr

    def scattering_factor(self, element):
        parameters = self._parameters[element]

        a = 3 * [0.02395 * element / (3 * (1 + parameters['V']))]
        a = a + 3 * [parameters['V'] * a[0]]
        b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

        f = lambda g: (a[0] * (1 - np.exp(-b[0] * g ** 2)) / g ** 2 +
                       a[1] * (1 - np.exp(-b[1] * g ** 2)) / g ** 2 +
                       a[2] * (1 - np.exp(-b[2] * g ** 2)) / g ** 2 +
                       a[3] * (1 - np.exp(-b[3] * g ** 2)) / g ** 2 +
                       a[4] * (1 - np.exp(-b[4] * g ** 2)) / g ** 2 +
                       a[5] * (1 - np.exp(-b[5] * g ** 2)) / g ** 2)

        dfdg = lambda g:1

        return f, dfdg
