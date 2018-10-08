import sys
from os import listdir

import numpy as np
from ase import Atoms
from ase import units
from ase.data import chemical_symbols
from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE
from scipy.optimize import basinhopping

from potentials import ParameterizedPotential
from utils import kappa
from potentials import load_parameters

cusp = lambda x, Z: np.pi / x ** 3 - Z / units.Bohr / x ** (5 / 2.)
cusp_derivative = lambda x, Z: - 4 * np.pi / x ** 4 + 5 / 2. * Z / units.Bohr / x ** (7 / 2.)


def linear_constraints_A(x, Z):
    return np.array([[2] * 5, [1 / bi for bi in x[5:]], [cusp(bi, Z) for bi in x[5:]]])


def linear_constraints_B(Z, f0):
    return np.array([f0, Z / (2 * np.pi ** 2 * units.Bohr), 0])


def get_jacobian(x, Z):
    return np.array([[2] * 5 + [0] * 5,
                     np.hstack((1 / x[5:], x[:5] * (- 1 / x[5:] ** 2))),
                     np.hstack((cusp(x[5:], Z), x[:5] * cusp_derivative(x[5:], Z)))])


def check_constraint(x, Z, f0):
    return linear_constraints_A(x, Z).dot(x[:5]) - linear_constraints_B(Z, f0)


def apply_linear_constraints(x, Z, f0):
    A = linear_constraints_A(x, Z)
    B = linear_constraints_B(Z, f0)

    B = B - np.sum(A[:, 3:] * x[3:5], axis=1)
    A = A[:, :3]

    x[:3] = np.linalg.solve(A, B)

    return x


class RandomDisplacementBounds(object):
    """random displacement with bounds"""

    def __init__(self, xmin, xmax, Z, f0, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize
        self.Z = Z
        self.f0 = f0

    def __call__(self, x):

        """take a random step but ensure the new position is within the bounds"""
        while True:

            # this is ugly, but it works for now...
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))

            xnew = apply_linear_constraints(xnew, self.Z, self.f0)

            if not np.any(np.isnan(xnew)):
                if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                    if np.all(np.abs(check_constraint(xnew, self.Z, self.f0)) < 1e-12):
                        break
        return xnew


def get_ae_potential(atoms, h=0.02):
    calc = GPAW(hund=True, eigensolver='cg', h=.1)
    atoms.set_calculator(calc)

    atoms.get_potential_energy()

    ps2ae = PS2AE(atoms.calc, h=h)
    return ps2ae.get_electrostatic_potential(ae=True, rcgauss=.02)


def get_scattering_factor(v, atoms):
    nx, ny, nz = v.shape
    lx, ly, lz = np.diag(atoms.cell)
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    v = np.roll(v, nx // 2, axis=0)
    v = np.roll(v, ny // 2, axis=1)
    v = np.roll(v, nz // 2, axis=2)

    f = - np.fft.fftn(v)[0, 0, :].real * (dx * dy * dz) * kappa
    g = np.fft.fftfreq(nx) / dx

    return f, g


def merge_parameter_files(name='GPAW.txt', folder='fit_gpaw_data/'):
    files = [f for f in listdir(folder) if f in [symbol + '.txt' for symbol in chemical_symbols]]

    with open(folder + name, 'w') as text_file:
        print('from GPAW', file=text_file)
        print('Z,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,rms_error', file=text_file)

        for f in files:
            par = load_parameters(folder + f)
            Z = list(par.keys())[0]
            par = [par[Z][key] for key in ('a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'rms_error')]
            print(','.join(map(str, [Z] + par)), file=text_file)


if __name__ == "__main__":
    Z = int(sys.argv[1])

    niter_basin = 10
    niter_local = 1000
    step_size = 20
    h = 0.02
    folder = 'fit_gpaw_data/'

    xmin = [-np.inf] * 5 + [1e-6] * 5
    xmax = [np.inf] * 10

    symbol = chemical_symbols[Z]

    atoms = Atoms(symbol, [(0, 0, 0)])
    atoms.center(vacuum=4)

    v = get_ae_potential(atoms, h=h)
    f, g = get_scattering_factor(v, atoms)

    print(g)
    sss
    f0 = f[0]
    f = f[:len(g) // 2]
    g = g[:len(g) // 2]

    x0 = [ParameterizedPotential(None, None, 'lobato').parameters[Z][key] for key in ('a1', 'a2', 'a3', 'a4', 'a5',
                                                                                      'b1', 'b2', 'b3', 'b4', 'b5')]
    x0 = apply_linear_constraints(x0, Z, f0)

    func = lambda a, b: (a[0] * (2 + b[0] * g ** 2) / (1 + b[0] * g ** 2) ** 2 +
                         a[1] * (2 + b[1] * g ** 2) / (1 + b[1] * g ** 2) ** 2 +
                         a[2] * (2 + b[2] * g ** 2) / (1 + b[2] * g ** 2) ** 2 +
                         a[3] * (2 + b[3] * g ** 2) / (1 + b[3] * g ** 2) ** 2 +
                         a[4] * (2 + b[4] * g ** 2) / (1 + b[4] * g ** 2) ** 2)

    error = lambda x: sum((func(x[:5], x[5:]) - f) ** 2)

    constraint_func = lambda x: linear_constraints_A(x, Z).dot(x[:5]) - linear_constraints_B(Z, f0)

    eq_constraint = {'type': 'eq',
                     'fun': constraint_func}

    take_step = RandomDisplacementBounds(xmin, xmax, Z, f0, step_size)

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    minimizer_kwargs = dict(method='SLSQP', bounds=bounds, constraints=[eq_constraint],
                            options={'maxiter': niter_local})

    result = basinhopping(error, x0, niter=niter_basin, minimizer_kwargs=minimizer_kwargs, take_step=take_step,
                          disp=True)

    assert np.all(np.abs(check_constraint(result.x, Z, f0)) < 1e-6)

    a, b = result.x[:5], result.x[5:]
    rms_error = np.sqrt(np.sum((func(a, b) - f) ** 2) / len(f))

    with open(folder + symbol + '.txt', 'w') as text_file:
        print('from GPAW', file=text_file)
        print('Z,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,rms_error', file=text_file)
        print(','.join(map(str, [Z] + list(result.x) + [rms_error])), file=text_file)
