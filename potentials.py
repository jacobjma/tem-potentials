import csv
import numbers

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
from scipy.special import erfc
from ase import units

import utils


def find_cutoff(func, min_value, xmin=1e-16, xmax=1000):
    return brentq(lambda x: func(x) - min_value, xmin, xmax)


def potential_spline_coeff(r_min, v_min, n_nodes, v, dvdr):
    r_cut = find_cutoff(v, v_min)

    r = utils.log_grid(r_min, r_cut, n_nodes)

    v = v(r) - v(r_cut) - (r - r_cut) * dvdr(r_cut)

    bc_left, bc_right = [(1, dvdr(r_min))], [(1, 0.)]

    return make_interp_spline(r, v, bc_type=(bc_left, bc_right))


def scattering_spline_coeff(g_cut, n_nodes, f, dfdg):
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


def squared_magnitude_grid(x, y, z):
    return (x ** 2)[:, None, None] + (y ** 2)[None, :, None] + (z ** 2)[None, None, :]


def lobato_potential(parameters):
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


def lobato_scattering_factor(parameters):
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


def lobato_density(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')]
    b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    rho = lambda r: 2 * np.pi ** 4 * units.Bohr * (a[0] / b[0] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[0])) +
                                                   a[1] / b[1] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[1])) +
                                                   a[2] / b[2] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[2])) +
                                                   a[3] / b[3] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[3])) +
                                                   a[4] / b[4] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[4])))

    return rho


def kirkland_potential(parameters):
    a = [np.pi * parameters[key] for key in ('a1', 'a2', 'a3')]
    b = [2 * np.pi * np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3')]
    c = [np.pi ** (3 / 2.) * parameters[key_c] / parameters[key_d] ** (3 / 2.) for key_c, key_d in
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


def kirkland_scattering_factor(parameters):
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


def gaussian_sum_potential(parameters):
    a = [np.pi ** (3 / 2.) * parameters[key_a] / (parameters[key_b] / 4) ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4'), ('b1', 'b2', 'b3', 'b4'))]

    b = [np.pi ** 2 / (parameters[key] / 4) for key in ('b1', 'b2', 'b3', 'b4')]

    v = lambda r: (a[0] * np.exp(-b[0] * r ** 2) +
                   a[1] * np.exp(-b[1] * r ** 2) +
                   a[2] * np.exp(-b[2] * r ** 2) +
                   a[3] * np.exp(-b[3] * r ** 2))

    dvdr = lambda r: (- 2 * a[0] * b[0] * r * np.exp(-b[0] * r ** 2)
                      - 2 * a[1] * b[1] * r * np.exp(-b[1] * r ** 2)
                      - 2 * a[2] * b[2] * r * np.exp(-b[2] * r ** 2)
                      - 2 * a[3] * b[3] * r * np.exp(-b[3] * r ** 2))

    return v, dvdr


def gaussian_sum_scattering_factor(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4')]
    b = [parameters[key] / 4 for key in ('b1', 'b2', 'b3', 'b4')]

    f = lambda g: (a[0] * np.exp(-b[0] * g ** 2) +
                   a[1] * np.exp(-b[1] * g ** 2) +
                   a[2] * np.exp(-b[2] * g ** 2) +
                   a[3] * np.exp(-b[3] * g ** 2))

    dfdg = lambda g: (- 2 * a[0] * b[0] * np.exp(-b[0] * g ** 2)
                      - 2 * a[1] * b[1] * np.exp(-b[1] * g ** 2)
                      - 2 * a[2] * b[2] * np.exp(-b[2] * g ** 2)
                      - 2 * a[3] * b[3] * np.exp(-b[3] * g ** 2))

    return f, dfdg


def weickenmeier_potential(parameters):
    a = 3 * [4 * np.pi * 0.02395 * parameters['Z'] / (3 * (1 + parameters['V']))]
    a = a + 3 * [parameters['V'] * a[0]]
    b = [np.pi / np.sqrt(parameters[key] / 4) for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

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


def weickenmeier_scattering_factor(parameters):
    a = 3 * [4 * 0.02395 * parameters['Z'] / (3 * (1 + parameters['V']))]
    a = a + 3 * [parameters['V'] * a[0]]
    b = [parameters[key] / 4 for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

    f = lambda g: (a[0] * (1 - np.exp(-b[0] * g ** 2)) / g ** 2 +
                   a[1] * (1 - np.exp(-b[1] * g ** 2)) / g ** 2 +
                   a[2] * (1 - np.exp(-b[2] * g ** 2)) / g ** 2 +
                   a[3] * (1 - np.exp(-b[3] * g ** 2)) / g ** 2 +
                   a[4] * (1 - np.exp(-b[4] * g ** 2)) / g ** 2 +
                   a[5] * (1 - np.exp(-b[5] * g ** 2)) / g ** 2)

    dfdg = lambda g: 1

    return f, dfdg


def load_parameters(filename):
    parameters = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        keys = next(reader)
        for _, row in enumerate(reader):
            values = list(map(float, row))
            parameters[int(row[0])] = dict(zip(keys, values))

    return parameters


potential_names = {'lobato': {'potential': lobato_potential,
                              'scattering_factor': lobato_scattering_factor,
                              'density': lobato_density,
                              'default_parameters': 'data/lobato.txt'},
                   'kirkland': {'potential': kirkland_potential,
                                'scattering_factor': kirkland_scattering_factor,
                                'density': None,
                                'default_parameters': 'data/kirkland.txt'},
                   'peng': {'potential': gaussian_sum_potential,
                            'scattering_factor': gaussian_sum_scattering_factor,
                            'density': None,
                            'default_parameters': 'data/peng.txt'},
                   'weickenmeier': {'potential': weickenmeier_potential,
                                    'scattering_factor': weickenmeier_scattering_factor,
                                    'density': None,
                                    'default_parameters': 'data/weickenmeier.txt'},
                   'gpaw': {'potential': kirkland_potential,
                            'scattering_factor': kirkland_scattering_factor,
                            'density': None,
                            'default_parameters': 'data/gpaw.txt'}
                   }


class ParameterizedPotential(object):

    def __init__(self, parametrization, atoms=None, grid=None, parameters=None):

        if atoms is None:
            self._atoms = None
        else:
            self.atoms = atoms

        if grid is None:
            self._grid = None
        elif isinstance(grid, numbers.Integral):
            self._grid = (grid,) * 3
        else:
            self._grid = grid

        self._potential = potential_names[parametrization]['potential']
        self._scattering_factor = potential_names[parametrization]['scattering_factor']

        self._density = potential_names[parametrization]['density']

        if parameters is None:
            self.parameters = load_parameters(potential_names[parametrization]['default_parameters'])
        elif isinstance(str, parameters):
            self.parameters = load_parameters(parameters)
        else:
            self.parameters = parameters

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if not utils.cell_is_rectangular(atoms.get_cell()):
            raise RuntimeError()
        self._atoms = atoms

    @property
    def grid(self):
        return self._grid

    @property
    def box(self):
        return np.diag(self._atoms.get_cell())

    def potential(self, number, return_derivative=False):
        if return_derivative:
            return self._potential(self.parameters[number])
        else:
            return self._potential(self.parameters[number])[0]

    def scattering_factor(self, number, return_derivative=False):
        if return_derivative:
            return self._scattering_factor(self.parameters[number])
        else:
            return self._scattering_factor(self.parameters[number])[0]

    def density(self, number):
        return self._density(self.parameters[number])

    def voxel_size(self):
        Lx, Ly, Lz = self.box
        Nx, Ny, Nz = self.grid
        return Lx / (Nx), Ly / (Ny), Lz / (Nz)

    def fourier_frequencies(self):
        Nx, Ny, Nz = self.grid
        gx = np.fft.fftfreq(Nx)
        gy = np.fft.fftfreq(Ny)
        gz = np.fft.fftfreq(Nz)
        return gx, gy, gz

    def spatial_frequencies(self):
        gx, gy, gz = self.fourier_frequencies()
        dx, dy, dz = self.voxel_size()
        return gx / dx, gy / dy, gz / dz

    def real_space_coordinates(self):

        x = np.linspace(0, self.box[0], self.grid[0], endpoint=False)
        y = np.linspace(0, self.box[1], self.grid[1], endpoint=False)
        z = np.linspace(0, self.box[2], self.grid[2], endpoint=False)

        r2 = squared_magnitude_grid(x, y, z)

        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        return x, y, z, r2

    def scattering_splines(self, g_cut, n_nodes):

        splines = {}
        for number in np.unique(self.atoms.get_atomic_numbers()):
            f, dfdg = self.scattering_factor(number, return_derivative=True)
            splines[number] = scattering_spline_coeff(g_cut, n_nodes, f, dfdg)

        return splines

    def calc_scattering(self, g_cut=None, n_nodes=50):

        dx, dy, dz = self.voxel_size()
        gx, gy, gz = self.fourier_frequencies()
        g = np.sqrt(squared_magnitude_grid(gx / dx, gy / dy, gz / dz))

        if g_cut is None:
            g_cut = np.sqrt((1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2)) / 4

        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')

        splines_lookup = self.scattering_splines(g_cut, n_nodes)
        f = np.zeros(self.grid, dtype='complex')
        for atom in self.atoms:
            splines = splines_lookup[atom.number]

            x = atom.x / dx
            y = atom.y / dy
            z = atom.z / dz

            #import matplotlib.pyplot as plt
            #plt.imshow(g[0])
            #plt.show()

            #sss np.exp(-2 * np.pi * 1j * (gx * x + gy * y + gz * z))

            f += splines(g) * utils.interpolated_translation(gx, gy, gz, x, y, z)

        return f

    def calc_from_scattering(self, g_cut=None, n_nodes=50):
        f = self.calc_scattering(g_cut, n_nodes)
        return np.fft.ifftn(f).real / np.prod(self.voxel_size()) / utils.kappa

    def potential_splines(self, r_min, v_min, n_nodes):

        if r_min is None:
            r_min = min(self.voxel_size()) * 2

        splines = {}
        for number in np.unique(self.atoms.get_atomic_numbers()):
            v, dvdr = self.potential(number, return_derivative=True)

            splines[number] = potential_spline_coeff(r_min, v_min, n_nodes, v, dvdr)

        return splines

    def repeated_positions(self, max_cutoff):

        #if np.any(max_cutoff > self.box):
        #    raise RuntimeError()

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

        return V / utils.kappa
