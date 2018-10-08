import numpy as np

import utils


def cut_reciprocal(x_cut):
    f = lambda x: 1 / x
    dfdx = lambda x: -1 / x ** 2
    return (lambda x: f(x) - f(x_cut) - (x - x_cut) * dfdx(x_cut))


def density2potential(n, atoms):
    if not utils.cell_is_rectangular(atoms.get_cell()):
        raise RuntimeError()

    Lx, Ly, Lz = np.diag(atoms.get_cell())
    Nx, Ny, Nz = n.shape
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    charge_ratio = np.sum(n) / np.sum(atoms.get_atomic_numbers())

    gx = np.fft.fftfreq(Nx)
    gy = np.fft.fftfreq(Ny)
    gz = np.fft.fftfreq(Nz)
    gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')

    f = np.fft.fftn(n)

    for atom in atoms:
        x, y, z = atom.position

        x /= dx
        y /= dy
        z /= dz

        f -= atom.number * charge_ratio * utils.interpolated_translation(gx, gy, gz, x, y, z)

    g2 = (gx / dx) ** 2 + (gy / dy) ** 2 + (gz / dz) ** 2
    # g2_max = (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2) / 4 * 2

    coeff = np.zeros((Nx, Ny, Nz))
    coeff[g2 != 0] = 1 / (2 ** 2 * np.pi ** 2 * g2[g2 != 0])  # cut_reciprocal(g2_max)(g2[g2 != 0])
    # coeff[g2 > g2_max] = 0
    f0 = f[0,0,0]
    f *= coeff
    f[0, 0, 0] = -f0

    v = np.fft.ifftn(f).real

    return v / utils.eps0

def density2scattering(n, atoms):
    if not utils.cell_is_rectangular(atoms.get_cell()):
        raise RuntimeError()

    Lx, Ly, Lz = np.diag(atoms.get_cell())
    Nx, Ny, Nz = n.shape
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    charge_ratio = np.sum(n) / np.sum(atoms.get_atomic_numbers())

    gx = np.fft.fftfreq(Nx)
    gy = np.fft.fftfreq(Ny)
    gz = np.fft.fftfreq(Nz)
    gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')

    f = np.fft.fftn(n)

    for atom in atoms:
        x, y, z = atom.position

        x /= dx
        y /= dy
        z /= dz

        f -= atom.number * charge_ratio * utils.interpolated_translation(gx, gy, gz, x, y, z)

    g2 = (gx / dx) ** 2 + (gy / dy) ** 2 + (gz / dz) ** 2
    # g2_max = (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2) / 4 * 2

    coeff = np.zeros((Nx, Ny, Nz))
    coeff[g2 != 0] = 1 / (2 ** 2 * np.pi ** 2 * g2[g2 != 0])  # cut_reciprocal(g2_max)(g2[g2 != 0])
    # coeff[g2 > g2_max] = 0

    f *= coeff

    return f