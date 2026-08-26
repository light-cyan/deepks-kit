import os
import sys
import math
from numbers import Real
import operator
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from time import time
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from deepks.data.force_schema import (
    ForceDataContract,
    ForceDataError,
    force_checkpoint_metadata,
    validate_force_data_contract,
    validate_force_checkpoint_metadata,
)
from deepks.gpu import DEFAULT_CUDA_DEVICE, require_cuda_device
from deepks.model.evaluate import (
    CorrectionPrediction,
    _predict_correction,
    _validate_model_state,
    model_reference,
)
from deepks.model.model import (
    CorrNet,
    FORCE_JACOBIAN_SEMANTICS,
    force_model_structure_evidence,
    model_execution_state_evidence,
    normalize_force_contract_fingerprint,
    validate_force_model_architecture,
)
from deepks.model.reader import (
    FORCE_MODE_DEEPHF_RELAXED,
    FORCE_MODE_NONE,
    GroupReader,
    _force_batch_error,
)
from deepks.utils import load_basis, load_dirs, load_elem_table


DEVICE = torch.device(DEFAULT_CUDA_DEVICE)


class ForceTrainingError(ValueError):
    """Raised when strict relaxed-force training data are incomplete or ambiguous."""


@dataclass(frozen=True)
class ErrorMetrics:
    """Unweighted elementwise error metrics for one prediction batch."""

    absolute_error_sum: torch.Tensor
    squared_error_sum: torch.Tensor
    count: int

    @property
    def mae(self) -> torch.Tensor:
        return self.absolute_error_sum / self.count

    @property
    def rmse(self) -> torch.Tensor:
        return torch.sqrt(self.squared_error_sum / self.count)


@dataclass(frozen=True)
class EvaluationResult:
    """Loss components, predictions, and separately reported physical metrics."""

    total_loss: torch.Tensor
    energy_loss: torch.Tensor
    force_loss: torch.Tensor | None
    energy_metrics: ErrorMetrics
    force_metrics: ErrorMetrics | None
    prediction: CorrectionPrediction


@dataclass(frozen=True)
class MetricSummary:
    """Dataset-level energy and optional force MAE/RMSE values."""

    energy_mae: float
    energy_rmse: float
    force_mae: float | None
    force_rmse: float | None


@dataclass(frozen=True)
class TrainingResult:
    """The trained model and its final train/validation metrics."""

    model: CorrNet
    training_metrics: MetricSummary
    validation_metrics: MetricSummary


def _error_metrics(predicted: torch.Tensor, target: torch.Tensor) -> ErrorMetrics:
    difference = (predicted - target).detach()
    return ErrorMetrics(
        absolute_error_sum=difference.abs().sum(),
        squared_error_sum=difference.square().sum(),
        count=difference.numel(),
    )


def force_contract_fingerprint(contract) -> str:
    """Return the normalized SHA-256 fingerprint of a force-data contract."""
    if not isinstance(contract, ForceDataContract):
        raise ForceTrainingError(
            "force-aware evaluation requires a validated ForceDataContract"
        )
    try:
        validate_force_data_contract(contract)
    except (ForceDataError, TypeError) as error:
        raise ForceTrainingError(f"invalid force-data contract: {error}") from error
    if contract.jacobian_semantics != FORCE_JACOBIAN_SEMANTICS:
        raise ForceTrainingError(
            "force-data contract must declare dq_dR_relaxed Jacobian semantics"
        )
    try:
        return normalize_force_contract_fingerprint(
            contract.force_contract_fingerprint
        )
    except (TypeError, ValueError) as error:
        raise ForceTrainingError(f"invalid force-data contract fingerprint: {error}") from error


def _force_contract_registry(value) -> tuple[ForceDataContract, ...]:
    if isinstance(value, ForceDataContract):
        contracts = (value,)
    elif isinstance(value, (tuple, list)) and value:
        contracts = tuple(value)
    else:
        raise ForceTrainingError(
            "force-aware evaluation requires validated ForceDataContract objects"
        )
    if any(not isinstance(contract, ForceDataContract) for contract in contracts):
        raise ForceTrainingError(
            "force-aware evaluation requires validated ForceDataContract objects"
        )
    fingerprint = force_contract_fingerprint(contracts[0])
    if any(force_contract_fingerprint(contract) != fingerprint for contract in contracts[1:]):
        raise ForceTrainingError(
            "force-data contract registry contains incompatible provenance"
        )
    return contracts


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _contract_projector_basis(contract: ForceDataContract):
    if not isinstance(contract, ForceDataContract):
        raise ForceTrainingError(
            "strict force training requires a validated ForceDataContract"
        )
    try:
        return contract.manifest["descriptor"]["projector_basis"]
    except (KeyError, TypeError) as error:
        raise ForceTrainingError(
            "force-data contract is missing canonical projector_basis metadata"
        ) from error


def validate_model_force_contract(model, contract: ForceDataContract) -> None:
    """Reject same-sized models whose projector or feature contract differs."""
    projector_basis = _contract_projector_basis(contract)
    expected_features = int(contract.dimensions["n_feature"])
    if getattr(model, "input_dim", None) != expected_features:
        raise ForceTrainingError(
            "model input dimension does not match the force-data contract: "
            f"{getattr(model, 'input_dim', None)} != {expected_features}"
        )
    if getattr(model, "elem_table", None) is not None:
        raise ForceTrainingError(
            "strict force models must encode the complete correction energy "
            "without an external element table"
        )
    if _metadata_signature(getattr(model, "_pbas", None)) != _metadata_signature(
        load_basis(projector_basis)
    ):
        raise ForceTrainingError(
            "model projector metadata does not match the force-data contract"
        )


def _require_sample_tensor(
    sample,
    name: str,
    *,
    ndim: int,
    check_finite: bool = True,
) -> torch.Tensor:
    if name not in sample:
        raise ValueError(f"sample is missing required field {name!r}")
    value = sample[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"sample field {name!r} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"sample field {name!r} must use torch.float64")
    if value.ndim != ndim:
        raise ValueError(
            f"sample field {name!r} must have rank {ndim}; received {tuple(value.shape)}"
        )
    if check_finite and not torch.isfinite(value).all():
        raise ValueError(f"sample field {name!r} must contain only finite values")
    return value


def _summarize_evaluations(results) -> MetricSummary:
    energy_absolute = energy_squared = 0.0
    force_absolute = force_squared = 0.0
    energy_count = force_count = 0
    for result in results:
        energy_absolute += result.energy_metrics.absolute_error_sum.item()
        energy_squared += result.energy_metrics.squared_error_sum.item()
        energy_count += result.energy_metrics.count
        if result.force_metrics is not None:
            force_absolute += result.force_metrics.absolute_error_sum.item()
            force_squared += result.force_metrics.squared_error_sum.item()
            force_count += result.force_metrics.count
    if energy_count == 0:
        raise ValueError("cannot summarize an empty evaluation")
    return MetricSummary(
        energy_mae=energy_absolute / energy_count,
        energy_rmse=math.sqrt(energy_squared / energy_count),
        force_mae=None if force_count == 0 else force_absolute / force_count,
        force_rmse=None if force_count == 0 else math.sqrt(force_squared / force_count),
    )


def evaluate_reader(model, reader, evaluator: "Evaluator") -> MetricSummary:
    """Evaluate one reader with the same predictor and metrics used for training."""
    def results():
        needs_gradient = (
            evaluator.force_factor > 0
            or evaluator.density_factor > 0
            or evaluator.gradient_penalty > 0
        )
        context = torch.enable_grad if needs_gradient else torch.no_grad
        for batch in reader.sample_all_batch():
            with context():
                yield evaluator.evaluate(model, batch, create_graph=False)

    return _summarize_evaluations(results())


def _training_batches(reader):
    """Yield exactly one bounded epoch, including exactly one single-frame batch."""
    group_batch = max(int(getattr(reader, "group_batch", 1)), 1)
    batch_size = int(reader.get_batch_size()) * group_batch
    if batch_size <= 0:
        raise ValueError("training batch size must be positive")
    batch_count = max(math.ceil(int(reader.get_train_size()) / batch_size), 1)
    sampler = (
        reader.sample_train
        if group_batch == 1
        else reader.sample_train_group
    )
    for _ in range(batch_count):
        yield sampler()


def fit_elem_const(g_reader, test_reader=None, elem_table=None, ridge_alpha=0.):
    if elem_table is None:
        elem_table = g_reader.compute_elem_const(ridge_alpha)
    elem_list, elem_const = elem_table
    g_reader.collect_elems(elem_list)
    g_reader.subtract_elem_const(elem_const)
    if test_reader is not None:
        test_reader.collect_elems(elem_list)
        test_reader.subtract_elem_const(elem_const)
    return elem_table


def preprocess(model, g_reader, 
                preshift=True, prescale=False, prescale_sqrt=False, prescale_clip=0,
                prefit=True, prefit_ridge=10, prefit_trainable=False):
    shift = model.input_shift.cpu().detach().numpy()
    scale = model.input_scale.cpu().detach().numpy()
    symmetry_sections = model.shell_sec # will be None if no embedding
    prefit_trainable = prefit_trainable and symmetry_sections is None # no embedding
    if preshift or prescale:
        descriptor_average, descriptor_std = g_reader.compute_data_stat(
            symmetry_sections
        )
        if preshift:
            shift = descriptor_average
        if prescale:
            scale = descriptor_std
            if prescale_sqrt: 
                scale = np.sqrt(scale)
            if prescale_clip: 
                scale = scale.clip(prescale_clip)
        model.set_normalization(shift, scale)
    if prefit:
        weight, bias = g_reader.compute_prefitting(
            shift=shift, scale=scale, 
            ridge_alpha=prefit_ridge, symm_sections=symmetry_sections)
        model.set_prefitting(weight, bias, trainable=prefit_trainable)


def make_loss(cap=None, shrink=None, reduction="mean"):
    def loss_fn(input, target):
        diff = target - input
        if shrink and shrink > 0:
            diff = F.softshrink(diff, shrink)
        sqdf = diff ** 2
        if cap and cap > 0:
            abdf = diff.abs()
            sqdf = torch.where(abdf < cap, sqdf, cap * (2*abdf - cap))
        if reduction is None or reduction.lower() == "none":
            return sqdf
        elif reduction.lower() == "mean":
            return sqdf.mean()
        elif reduction.lower() == "sum":
            return sqdf.sum()
        elif reduction.lower() in ("batch", "bmean"):
            return sqdf.sum() / sqdf.shape[0]
        else:
            raise ValueError(f"{reduction} is not a valid reduction type")
    return loss_fn

# equiv to nn.MSELoss()
L2LOSS = make_loss(cap=None, shrink=None, reduction="mean")


class Evaluator:
    def __init__(self,
                 energy_factor=1., force_factor=0., 
                 density_factor=0., grad_penalty=0., 
                 energy_lossfn=None, force_lossfn=None,
                 force_contract=None):
        for name, value in (
            ("energy_factor", energy_factor),
            ("force_factor", force_factor),
            ("density_factor", density_factor),
            ("grad_penalty", grad_penalty),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        # energy term
        if energy_lossfn is None:
            energy_lossfn = {}
        if isinstance(energy_lossfn, dict):
            energy_lossfn = make_loss(**energy_lossfn)
        self.energy_factor = energy_factor
        self.energy_lossfn = energy_lossfn
        # force term
        if force_lossfn is None:
            force_lossfn = {}
        if isinstance(force_lossfn, dict):
            force_lossfn = make_loss(**force_lossfn)
        self.force_factor = force_factor
        self.force_lossfn = force_lossfn
        # Coulomb-loss term; requires the energy gradient with respect to descriptors.
        self.density_factor = density_factor
        # gradient penalty, not very useful
        self.gradient_penalty = grad_penalty
        if self.force_factor == 0 and force_contract is not None:
            raise ForceTrainingError(
                "a force-data contract requires force_factor > 0"
            )
        self.force_contracts = (
            _force_contract_registry(force_contract)
            if self.force_factor > 0
            else ()
        )
        self.force_contract = self.force_contracts[0] if self.force_contracts else None
        self._validated_model = None
        self._model_structure_evidence = None
        self._model_state_evidence = None
        self._model_device = None

    def _validate_model(self, model, descriptor) -> None:
        structure = force_model_structure_evidence(model) if self.force_factor > 0 else None
        if model is not self._validated_model or structure != self._model_structure_evidence:
            if self.force_factor > 0:
                validate_force_model_architecture(model, training=model.training)
                validate_model_force_contract(model, self.force_contract)
                structure = force_model_structure_evidence(model)
            _validate_model_state(model, descriptor)
            self._validated_model = model
            self._model_structure_evidence = structure
            self._model_state_evidence = model_execution_state_evidence(model)
            self._model_device = descriptor.device
            return
        state = model_execution_state_evidence(model)
        if state != self._model_state_evidence or descriptor.device != self._model_device:
            _validate_model_state(model, descriptor)
            self._model_state_evidence = state
            self._model_device = descriptor.device

    def __call__(self, model, sample):
        return self.evaluate(model, sample, create_graph=True).total_loss

    def evaluate(self, model, sample, *, create_graph=False) -> EvaluationResult:
        """Evaluate loss components and separate energy/force error metrics."""
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping")
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be bool")
        strict_sample_fields = {
            "force",
            "dq_dR_relaxed",
        }
        if self.force_factor == 0 and strict_sample_fields.intersection(sample):
            raise ForceTrainingError(
                "strict force samples cannot be evaluated through an energy-only path"
            )
        reference = model_reference(model)
        energy = _require_sample_tensor(sample, "energy", ndim=2)
        descriptor = _require_sample_tensor(sample, "descriptor", ndim=3)
        if energy.shape != (descriptor.shape[0], 1):
            raise ValueError(
                "sample energy must have shape (frame, 1) matching descriptor; "
                f"received {tuple(energy.shape)} and {tuple(descriptor.shape)}"
            )
        energy = energy.to(device=reference.device, non_blocking=True)
        descriptor = descriptor.to(device=reference.device, non_blocking=True)
        self._validate_model(model, descriptor)

        target_force = None
        relaxed_jacobian = None
        if self.force_factor > 0:
            target_force = _require_sample_tensor(sample, "force", ndim=3)
            relaxed_jacobian = _require_sample_tensor(
                sample,
                "dq_dR_relaxed",
                ndim=5,
                check_finite=False,
            )
            batch_error = _force_batch_error(sample, self.force_contracts)
            if batch_error is not None:
                raise ForceTrainingError(batch_error)
            expected_force_shape = (
                descriptor.shape[0],
                relaxed_jacobian.shape[1],
                3,
            )
            if target_force.shape != expected_force_shape:
                raise ValueError(
                    "sample force shape must exactly match the relaxed-Jacobian raw-atom "
                    f"axis; expected {expected_force_shape}, received {tuple(target_force.shape)}"
                )
            target_force = target_force.to(
                device=reference.device,
                non_blocking=True,
            )
            relaxed_jacobian = relaxed_jacobian.to(
                device=reference.device,
                non_blocking=True,
            )

        needs_auxiliary_gradient = (
            self.density_factor > 0 or self.gradient_penalty > 0
        )
        if needs_auxiliary_gradient and self.force_factor == 0:
            descriptor = descriptor.detach().requires_grad_(True)
        prediction = _predict_correction(
            model,
            descriptor,
            dq_dR_relaxed=relaxed_jacobian,
            require_force=self.force_factor > 0,
            create_graph=create_graph,
        )
        energy_loss = self.energy_lossfn(prediction.energy, energy)
        if not isinstance(energy_loss, torch.Tensor) or energy_loss.numel() != 1:
            raise ValueError("energy loss function must return one scalar tensor")
        total_loss = self.energy_factor * energy_loss
        energy_metrics = _error_metrics(prediction.energy, energy)

        force_loss = None
        force_metrics = None
        if self.force_factor > 0:
            force_loss = self.force_lossfn(prediction.force, target_force)
            if not isinstance(force_loss, torch.Tensor) or force_loss.numel() != 1:
                raise ValueError("force loss function must return one scalar tensor")
            total_loss = total_loss + self.force_factor * force_loss
            force_metrics = _error_metrics(prediction.force, target_force)

        energy_descriptor_gradient = prediction.descriptor_gradient
        if needs_auxiliary_gradient and energy_descriptor_gradient is None:
            if prediction.energy.requires_grad:
                (energy_descriptor_gradient,) = torch.autograd.grad(
                    prediction.energy,
                    descriptor,
                    grad_outputs=torch.ones_like(prediction.energy),
                    retain_graph=create_graph,
                    create_graph=create_graph,
                    only_inputs=True,
                    allow_unused=True,
                )
            if energy_descriptor_gradient is None:
                energy_descriptor_gradient = torch.zeros_like(descriptor)

        if self.gradient_penalty > 0:
            reference_orbital_gradient = _require_sample_tensor(
                sample,
                "reference_orbital_gradient",
                ndim=2,
            ).to(device=reference.device, non_blocking=True)
            descriptor_orbital_gradient_jacobian = _require_sample_tensor(
                sample,
                "descriptor_orbital_gradient_jacobian",
                ndim=4,
            ).to(device=reference.device, non_blocking=True)
            total_orbital_gradient = torch.einsum(
                "...apg,...ap->...g",
                descriptor_orbital_gradient_jacobian,
                energy_descriptor_gradient,
            ) + reference_orbital_gradient
            total_loss = total_loss + self.gradient_penalty * (
                total_orbital_gradient.pow(2).mean(0).sum()
            )

        if self.density_factor > 0:
            coulomb_loss_descriptor_gradient = _require_sample_tensor(
                sample,
                "coulomb_loss_descriptor_gradient",
                ndim=3,
            ).to(device=reference.device, non_blocking=True)
            if coulomb_loss_descriptor_gradient.shape != descriptor.shape:
                raise ValueError(
                    "coulomb_loss_descriptor_gradient shape must match descriptor"
                )
            total_loss = total_loss + self.density_factor * (
                coulomb_loss_descriptor_gradient * energy_descriptor_gradient
            ).mean(0).sum()

        if total_loss.numel() != 1 or not torch.isfinite(total_loss).all():
            raise ValueError("total training loss must be one finite scalar")
        return EvaluationResult(
            total_loss=total_loss,
            energy_loss=energy_loss,
            force_loss=force_loss,
            energy_metrics=energy_metrics,
            force_metrics=force_metrics,
            prediction=prediction,
        )


def _training_integer(value, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _training_real(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def train(model, g_reader, n_epoch=1000, test_reader=None, *,
          energy_factor=1., force_factor=0., density_factor=0.,
          energy_loss=None, force_loss=None, grad_penalty=0.,
          start_lr=0.001, decay_steps=100, decay_rate=0.96, stop_lr=None,
          weight_decay=0.,  fix_embedding=False,
          display_epoch=100, ckpt_file="model.pth", device=DEVICE,
          force_contract=None):

    n_epoch = _training_integer(n_epoch, "n_epoch", allow_zero=True)
    decay_steps = _training_integer(decay_steps, "decay_steps")
    display_epoch = _training_integer(display_epoch, "display_epoch")
    start_lr = _training_real(start_lr, "start_lr")
    decay_rate = _training_real(decay_rate, "decay_rate")
    weight_decay = _training_real(weight_decay, "weight_decay", allow_zero=True)
    energy_factor = _training_real(energy_factor, "energy_factor", allow_zero=True)
    force_factor = _training_real(force_factor, "force_factor", allow_zero=True)
    density_factor = _training_real(density_factor, "density_factor", allow_zero=True)
    grad_penalty = _training_real(grad_penalty, "grad_penalty", allow_zero=True)
    if stop_lr is not None:
        stop_lr = _training_real(stop_lr, "stop_lr")
        if n_epoch < decay_steps:
            raise ValueError("stop_lr requires at least one scheduled decay step")

    device = require_cuda_device(device)
    model = model.to(device)
    model.eval()
    print("# working on device:", device)
    if test_reader is None:
        test_reader = g_reader
    training_reader_contract = getattr(g_reader, "force_contract", None)
    validation_reader_contract = getattr(test_reader, "force_contract", None)
    if force_factor == 0 and (
        training_reader_contract is not None
        or validation_reader_contract is not None
    ):
        raise ForceTrainingError(
            "strict force-data readers cannot be trained through an energy-only path"
        )
    if force_factor > 0:
        if force_contract is None:
            force_contract = training_reader_contract
        training_fingerprint = force_contract_fingerprint(force_contract)
        test_contract = validation_reader_contract or force_contract
        test_fingerprint = force_contract_fingerprint(test_contract)
        if test_fingerprint != training_fingerprint:
            raise ForceTrainingError(
                "training and validation force-data contracts do not match"
            )
        if isinstance(force_contract, ForceDataContract):
            if not isinstance(test_contract, ForceDataContract):
                raise ForceTrainingError(
                    "validation data do not expose a validated ForceDataContract"
                )
            validate_force_checkpoint_metadata(
                force_checkpoint_metadata(test_contract),
                force_contract,
            )
            validate_model_force_contract(model, force_contract)
        training_contracts = getattr(
            g_reader,
            "force_contracts",
            (force_contract,),
        )
        validation_contracts = getattr(
            test_reader,
            "force_contracts",
            (test_contract,),
        )
    else:
        training_contracts = None
        validation_contracts = None
    # fix parameters if needed
    if fix_embedding and model.embedder is not None:
        model.embedder.requires_grad_(False)
    # set up optimizer and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    if stop_lr is not None:
        decay_rate = (stop_lr / start_lr) ** (1 / (n_epoch // decay_steps))
        print(f"# resetting decay_rate: {decay_rate:.4f} "
              + f"to satisfy stop_lr: {stop_lr:.2e}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    # make evaluators for training
    evaluator = Evaluator(energy_factor=energy_factor, force_factor=force_factor, 
                          energy_lossfn=energy_loss, force_lossfn=force_loss,
                          density_factor=density_factor, grad_penalty=grad_penalty,
                          force_contract=training_contracts)
    # Validation uses the same energy/relaxed-force predictor and reports each
    # physical metric independently of the training loss weights.
    test_eval = Evaluator(
        energy_factor=1.,
        energy_lossfn=L2LOSS,
        force_factor=1. if force_factor > 0 else 0.,
        force_lossfn=L2LOSS,
        density_factor=0.,
        grad_penalty=0.,
        force_contract=validation_contracts,
    )

    if force_factor > 0:
        print(
            "# epoch      trn_e_rmse tst_e_rmse trn_f_rmse tst_f_rmse "
            "       lr  trn_time  tst_time"
        )
    else:
        print("# epoch      trn_e_rmse tst_e_rmse        lr  trn_time  tst_time")
    tic = time()
    training_metrics = evaluate_reader(model, g_reader, evaluator)
    validation_metrics = evaluate_reader(model, test_reader, test_eval)
    tst_time = time() - tic
    if force_factor > 0:
        print(
            f"  {0:<8d}  {training_metrics.energy_rmse:>.2e}  "
            f"{validation_metrics.energy_rmse:>.2e}  "
            f"{training_metrics.force_rmse:>.2e}  "
            f"{validation_metrics.force_rmse:>.2e}  "
            f"{start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}"
        )
    else:
        print(
            f"  {0:<8d}  {training_metrics.energy_rmse:>.2e}  "
            f"{validation_metrics.energy_rmse:>.2e}  "
            f"{start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}"
        )

    for epoch in range(1, n_epoch+1):
        tic = time()
        for sample in _training_batches(g_reader):
            model.train()
            optimizer.zero_grad()
            loss = evaluator(model, sample)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % display_epoch == 0:
            model.eval()
            trn_time = time() - tic
            tic = time()
            training_metrics = evaluate_reader(model, g_reader, evaluator)
            validation_metrics = evaluate_reader(model, test_reader, test_eval)
            tst_time = time() - tic
            if force_factor > 0:
                print(
                    f"  {epoch:<8d}  {training_metrics.energy_rmse:>.2e}  "
                    f"{validation_metrics.energy_rmse:>.2e}  "
                    f"{training_metrics.force_rmse:>.2e}  "
                    f"{validation_metrics.force_rmse:>.2e}  "
                    f"{scheduler.get_last_lr()[0]:>.2e}  "
                    f"{trn_time:>8.2f}  {tst_time:8.2f}"
                )
            else:
                print(
                    f"  {epoch:<8d}  {training_metrics.energy_rmse:>.2e}  "
                    f"{validation_metrics.energy_rmse:>.2e}  "
                    f"{scheduler.get_last_lr()[0]:>.2e}  "
                    f"{trn_time:>8.2f}  {tst_time:8.2f}"
                )
            if ckpt_file:
                extra_info = {}
                if force_factor > 0:
                    extra_info["force_training"] = force_checkpoint_metadata(
                        force_contract
                    )
                model.save(ckpt_file, **extra_info)

    if n_epoch == 0 or n_epoch % display_epoch != 0:
        model.eval()
        training_metrics = evaluate_reader(model, g_reader, evaluator)
        validation_metrics = evaluate_reader(model, test_reader, test_eval)
    if ckpt_file:
        extra_info = {}
        if force_factor > 0:
            extra_info["force_training"] = force_checkpoint_metadata(force_contract)
        model.save(ckpt_file, **extra_info)
    return TrainingResult(
        model=model,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
    )
    

def main(train_paths, test_paths=None,
         restart=None, ckpt_file=None, 
         model_args=None, data_args=None, 
         preprocess_args=None, train_args=None, 
         projector_basis=None, fit_elem=False,
         seed=None, device=None):
    selected_device = require_cuda_device(device)
    if seed is None: 
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_args = {} if model_args is None else dict(model_args)
    data_args = {} if data_args is None else dict(data_args)
    preprocess_args = {} if preprocess_args is None else dict(preprocess_args)
    train_args = {} if train_args is None else dict(train_args)
    if projector_basis is not None:
        model_args["proj_basis"] = projector_basis
    if ckpt_file is not None:
        train_args["ckpt_file"] = ckpt_file
    train_args["device"] = selected_device

    force_enabled = train_args.get("force_factor", 0.) > 0
    configured_force_mode = data_args.get("force_mode", FORCE_MODE_NONE)
    if force_enabled:
        if fit_elem:
            raise ForceTrainingError(
                "strict force training does not support external element fitting"
            )
        if configured_force_mode not in (
            FORCE_MODE_NONE,
            FORCE_MODE_DEEPHF_RELAXED,
        ):
            raise ForceTrainingError(
                "force_factor > 0 conflicts with the configured data force_mode"
            )
        if "force_mode" in data_args and configured_force_mode == FORCE_MODE_NONE:
            raise ForceTrainingError(
                "force_factor > 0 requires force_mode='deephf_relaxed'"
            )
        data_args["force_mode"] = FORCE_MODE_DEEPHF_RELAXED
    elif configured_force_mode == FORCE_MODE_DEEPHF_RELAXED:
        raise ForceTrainingError(
            "force_mode='deephf_relaxed' requires force_factor > 0"
        )

    train_paths = load_dirs(train_paths)
    # print(f'# training with {len(train_paths)} system(s)')
    g_reader = GroupReader(train_paths, **data_args)
    if test_paths is not None:
        test_paths = load_dirs(test_paths)
        # print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    force_contract = getattr(g_reader, "force_contract", None)
    if force_enabled:
        projector_basis_from_contract = _contract_projector_basis(force_contract)
        if model_args.get("proj_basis") is None:
            model_args["proj_basis"] = projector_basis_from_contract
    if restart is not None:
        expected_fingerprint = (
            force_contract_fingerprint(force_contract)
            if force_enabled
            else None
        )
        model = CorrNet.load(
            restart,
            require_force_metadata=force_enabled,
            expected_force_contract_fingerprint=expected_fingerprint,
            expected_force_contract=force_contract if force_enabled else None,
        )
        if force_enabled:
            validate_model_force_contract(model, force_contract)
        elif model.elem_table is not None:
            fit_elem_const(g_reader, test_reader, model.elem_table)
    else:
        input_dim = g_reader.descriptor_size
        if model_args.get("input_dim", input_dim) != input_dim:
            print(f"# `input_dim` in `model_args` does not match data",
                  f"({input_dim}).", "Use the one in data.", file=sys.stderr)
        model_args["input_dim"] = input_dim
        if fit_elem:
            elem_table = model_args.get("elem_table", None)
            if isinstance(elem_table, str):
                elem_table = load_elem_table(elem_table)
            elem_table = fit_elem_const(g_reader, test_reader, elem_table)
            model_args["elem_table"] = elem_table
        model = CorrNet(**model_args).double()
        if force_enabled:
            validate_model_force_contract(model, force_contract)
        
    model = model.to(selected_device)
    preprocess(model, g_reader, **preprocess_args)
    return train(
        model,
        g_reader,
        test_reader=test_reader,
        force_contract=force_contract,
        **train_args,
    )


if __name__ == "__main__":
    from deepks.main import train_cli as cli
    cli()
