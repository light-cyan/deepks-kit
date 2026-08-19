"""Optional DeePKS electronic-gradient and density-loss operations."""

import time

import numpy as np
import torch
from pyscf.lib import logger

from deepks.descriptor import descriptor as torch_descriptor
from deepks.descriptor import occupied_virtual_gradient, spin_summed_ao_density


def descriptor_orbital_gradient_jacobian(
    method,
    mo_coeff=None,
    mo_occ=None,
    ao_jacobian=None,
):
    """Return the descriptor Jacobian in occupied-virtual coordinates."""
    if mo_occ is None:
        mo_occ = method.mo_occ
    if mo_coeff is None:
        mo_coeff = method.mo_coeff
    if ao_jacobian is None:
        density = method.make_rdm1(mo_coeff, mo_occ)
        ao_jacobian = method.dq_dP(density)
    if mo_coeff.ndim >= 3 and mo_occ.ndim >= 2:
        return np.concatenate(
            [
                descriptor_orbital_gradient_jacobian(
                    method,
                    spin_coefficients,
                    spin_occupations,
                    ao_jacobian,
                )
                for spin_coefficients, spin_occupations in zip(mo_coeff, mo_occ)
            ],
            axis=-1,
        )
    occupied = mo_occ > 0
    occupations = torch.from_numpy(mo_occ[occupied]).to(method.device)
    occupied_coefficients = torch.from_numpy(mo_coeff[:, occupied]).to(
        method.device
    )
    virtual_coefficients = torch.from_numpy(mo_coeff[:, ~occupied]).to(
        method.device
    )
    operators = torch.from_numpy(ao_jacobian).to(method.device)
    return occupied_virtual_gradient(
        operators,
        virtual_coefficients,
        occupied_coefficients,
        occupations,
    ).cpu().numpy()


def coulomb_loss(method, fock=None, overlap=None, mo_occ=None):
    """Build the Coulomb density-loss function and its AO derivative."""
    nao = method.mol.nao
    fock = (
        fock if fock is not None else method.get_fock()
    ).reshape(-1, nao, nao)
    overlap = overlap if overlap is not None else method.get_ovlp()
    mo_occ = (
        mo_occ if mo_occ is not None else method.mo_occ
    ).reshape(-1, nao)

    def evaluate(potential, target_density):
        loss_sum = 0.0
        gradient_sum = 0.0
        target_density = target_density.reshape(fock.shape)
        for target, reference_fock, occupations in zip(
            target_density,
            fock,
            mo_occ,
        ):
            occupied = occupations > 0
            orbital_energy, orbital_coefficients = method._eigh(
                reference_fock + potential,
                overlap,
            )
            occupied_energy = orbital_energy[occupied]
            virtual_energy = orbital_energy[~occupied]
            occupied_coefficients = orbital_coefficients[:, occupied]
            virtual_coefficients = orbital_coefficients[:, ~occupied]
            density = (
                occupied_coefficients * occupations[occupied]
            ) @ occupied_coefficients.T
            density_difference = density - target
            coulomb_potential = method.get_j(dm=density_difference)
            loss_sum += 0.5 * np.einsum(
                "ij,ji",
                density_difference,
                coulomb_potential,
            )
            energy_denominator = 1.0 / (
                -virtual_energy.reshape(-1, 1) + occupied_energy
            )
            transformed = (
                virtual_coefficients.T
                @ coulomb_potential
                @ occupied_coefficients
                * occupations[occupied]
                * energy_denominator
            )
            derivative = (
                virtual_coefficients @ transformed @ occupied_coefficients.T
            )
            gradient_sum += derivative + derivative.T
        return loss_sum, gradient_sum

    return evaluate


def coulomb_loss_descriptor_gradient(method, target_density):
    """Return the Coulomb-loss gradient with respect to descriptor values."""
    loss_function = coulomb_loss(method)
    density = spin_summed_ao_density(method.make_rdm1())
    tensor_density = torch.from_numpy(density).requires_grad_()
    values = torch_descriptor(
        tensor_density,
        method._descriptor.overlap_shells,
    ).requires_grad_()
    _, density_loss_gradient = loss_function(
        np.zeros_like(density),
        target_density,
    )
    descriptor_potential = torch.zeros_like(values).requires_grad_()
    (ao_potential,) = torch.autograd.grad(
        values,
        tensor_density,
        descriptor_potential,
        create_graph=True,
    )
    (result,) = torch.autograd.grad(
        ao_potential,
        descriptor_potential,
        torch.from_numpy(density_loss_gradient),
    )
    return result.detach().cpu().numpy()


def optimize_descriptor_potential(
    method,
    target_density,
    target_correction_gradient=None,
    descriptor_jacobian=None,
    nstep=1,
    force_factor=1.0,
    **optimizer_arguments,
):
    """Optimize a descriptor potential against density and force targets."""
    loss_function = coulomb_loss(
        method,
        fock=method.get_fock(vhf=method.reference_effective_potential()),
    )
    density = spin_summed_ao_density(method.make_rdm1())
    tensor_density = torch.from_numpy(density).requires_grad_()
    values = torch_descriptor(
        tensor_density,
        method._descriptor.overlap_shells,
    ).requires_grad_()
    correction_energy = method.model(values.to(method.device))
    (descriptor_potential,) = torch.autograd.grad(
        correction_energy,
        values,
    )
    descriptor_potential = descriptor_potential.requires_grad_()
    target_gradient = (
        torch.from_numpy(target_correction_gradient)
        if target_correction_gradient is not None
        else None
    )
    coordinate_jacobian = (
        torch.from_numpy(descriptor_jacobian)
        if descriptor_jacobian is not None
        else None
    )

    def closure():
        (ao_potential,) = torch.autograd.grad(
            values,
            tensor_density,
            descriptor_potential,
            retain_graph=True,
            create_graph=True,
        )
        loss, density_loss_gradient = loss_function(
            ao_potential.detach().numpy(),
            target_density,
        )
        gradient = torch.autograd.grad(
            ao_potential,
            descriptor_potential,
            torch.from_numpy(density_loss_gradient),
            only_inputs=True,
        )[0]
        if target_gradient is not None and coordinate_jacobian is not None:
            predicted_gradient = torch.tensordot(
                coordinate_jacobian,
                descriptor_potential,
            )
            force_loss = force_factor * torch.sum(
                (predicted_gradient - target_gradient) ** 2
            )
            gradient += torch.autograd.grad(
                force_loss,
                descriptor_potential,
                only_inputs=True,
            )[0]
            loss += force_loss
        descriptor_potential.grad = gradient
        return loss

    optimizer = torch.optim.LBFGS(
        [descriptor_potential],
        **optimizer_arguments,
    )
    timer = (time.process_time(), time.perf_counter())
    for _ in range(nstep):
        optimizer.step(closure)
        timer = logger.timer(method, "LBFGS step", *timer)
    logger.note(method, "optimized descriptor-potential loss = %s", closure())
    return descriptor_potential.detach().numpy()


def optimize_descriptor_potential_from_gradient(
    gradient,
    target_density,
    target_gradient,
    nstep=1,
    force_factor=1.0,
    **optimizer_arguments,
):
    """Optimize a descriptor potential using a DeePKS gradient result."""
    target_correction_gradient = target_gradient - gradient.reference_gradient()
    descriptor_jacobian = gradient.dq_dR_explicit()
    return optimize_descriptor_potential(
        gradient.base,
        target_density=target_density,
        target_correction_gradient=target_correction_gradient,
        descriptor_jacobian=descriptor_jacobian,
        nstep=nstep,
        force_factor=force_factor,
        **optimizer_arguments,
    )
