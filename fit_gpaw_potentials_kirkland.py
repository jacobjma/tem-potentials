import sys
from os import listdir

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE
from scipy.optimize import basinhopping

from potentials import ParameterizedPotential
from utils import kappa
from potentials import load_parameters


class RandomDisplacementBounds(object):
    """random displacement with bounds"""

    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):

        """take a random step but ensure the new position is within the bounds"""
        while True:

            # this is ugly, but it works for now...
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))

            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew


def get_ae_potential(atoms):
    calc = GPAW(hund=True, eigensolver='cg', h=.2)
    atoms.set_calculator(calc)

    atoms.get_potential_energy()

    ps2ae = PS2AE(atoms.calc)
    return ps2ae.get_electrostatic_potential(ae=True)


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
        print('Z,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,rms_error', file=text_file)

        for f in files:
            par = load_parameters(folder + f)
            Z = list(par.keys())[0]
            par = [par[Z][key] for key in
                   ('a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'rms_error')]
            print(','.join(map(str, [Z] + par)), file=text_file)


if __name__ == "__main__":
    Z = int(sys.argv[1])

    niter_basin = 1
    niter_local = 1000
    step_size = .05
    folder = 'fit_gpaw_data/'

    xmin = [0] * 12
    xmax = [25] * 12

    symbol = chemical_symbols[Z]

    atoms = Atoms(symbol, [(0, 0, 0)], pbc=True)
    atoms.center(vacuum=7)

    v = get_ae_potential(atoms)
    f, g = get_scattering_factor(v, atoms)

    kirkland = ParameterizedPotential('kirkland')

    f0 = kirkland.scattering_factor(Z)(0)

    f = f[0:len(g) // 2]
    g = g[0:len(g) // 2]
    f[0] = f0

    x0 = [kirkland.parameters[Z][key] for key in
          ('a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3')]

    func = lambda a, b, c, d: (a[0] / (b[0] + g ** 2) + c[0] * np.exp(-d[0] * g ** 2) +
                               a[1] / (b[1] + g ** 2) + c[1] * np.exp(-d[1] * g ** 2) +
                               a[2] / (b[2] + g ** 2) + c[2] * np.exp(-d[2] * g ** 2))

    dg = g[1] - g[0]
    min_weight = .05
    weights = (np.log((g + dg) / dg) + min_weight)

    if np.any(np.isnan(weights)):
        raise RuntimeError('nan in weights')

    error = lambda x: sum((func(x[:3], x[3:6], x[6:9], x[9:12]) - f) ** 2 / weights ** 2)

    take_step = RandomDisplacementBounds(xmin, xmax, step_size)

    bounds = [(low, high) for low, high in zip(xmin, xmax)]

    minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds, options={'maxiter': niter_local})

    result = basinhopping(error, x0, niter=niter_basin, minimizer_kwargs=minimizer_kwargs, take_step=take_step,
                          disp=True)

    a, b, c, d = result.x[:3], result.x[3:6], result.x[6:9], result.x[9:12]
    rms_error = np.sqrt(np.sum((func(a, b, c, d) - f) ** 2) / len(f))

    with open(folder + symbol + '.txt', 'w') as text_file:
        print('from GPAW', file=text_file)
        print('Z,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,rms_error', file=text_file)
        print(','.join(map(str, [Z] + list(result.x) + [rms_error])), file=text_file)
