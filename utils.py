import numpy as np

from ase import units

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

wavelength_from_energy = lambda x: (
        units._hplanck * units._c / np.sqrt(x * (2 * units._me * units._c ** 2 / units._e + x)) / units._e * 1e10)

sigma_from_energy = lambda x: (
        2 * np.pi * units._me * units.kg * units._e * units.C * wavelength_from_energy(x) / (
        units._hplanck * units.s * units.J) ** 2)


def log_grid(start, stop, num):
    dt = np.log(stop / start) / (num - 1)
    return start * np.exp(dt * np.linspace(0, num - 1, num))


def cell_is_rectangular(cell, tol=1e-14):
    return np.all(np.abs(cell[~np.eye(cell.shape[0], dtype=bool)]) < tol)


def _nearest_neighbor(x):
    x = float(x)

    if (x).is_integer():
        xa, xb = x, 0
        dxa, dxb = 1, 0
    else:
        xa, xb = np.floor(x), np.ceil(x)
        dxa, dxb = x - xa, xb - x

    return xa, xb, dxa, dxb


def interpolated_translation(gx, gy, gz, x, y, z):
    xa, xb, dxa, dxb = _nearest_neighbor(x)
    ya, yb, dya, dyb = _nearest_neighbor(y)
    za, zb, dza, dzb = _nearest_neighbor(z)

    f = (dxa * dya * dza * np.exp(2 * np.pi * 1j * (gx * xa + gy * ya + gz * za)) +
         dxb * dya * dza * np.exp(2 * np.pi * 1j * (gx * xb + gy * ya + gz * za)) +
         dxa * dyb * dza * np.exp(2 * np.pi * 1j * (gx * xa + gy * yb + gz * za)) +
         dxa * dya * dzb * np.exp(2 * np.pi * 1j * (gx * xa + gy * ya + gz * zb)) +
         dxb * dyb * dza * np.exp(2 * np.pi * 1j * (gx * xb + gy * yb + gz * za)) +
         dxb * dya * dzb * np.exp(2 * np.pi * 1j * (gx * xb + gy * ya + gz * zb)) +
         dxa * dyb * dzb * np.exp(2 * np.pi * 1j * (gx * xa + gy * yb + gz * zb)) +
         dxb * dyb * dzb * np.exp(2 * np.pi * 1j * (gx * xb + gy * yb + gz * zb)))

    return f
