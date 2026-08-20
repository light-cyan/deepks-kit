import math
import inspect
from collections.abc import Mapping
import numpy as np
import torch
import torch.nn as nn 
from torch.nn import functional as F
from deepks.utils import load_basis, get_shell_sec
from deepks.utils import load_elem_table

SCALE_EPS = 1e-8
CHECKPOINT_FORMAT_VERSION = 1
FORCE_JACOBIAN_SEMANTICS = "dq_dR_relaxed"
FORCE_CHECKPOINT_METADATA_KEYS = {
    "schema_id",
    "schema_version",
    "compatibility_fingerprint",
    "jacobian_semantics",
    "n_feature",
    "descriptor_definition",
    "descriptor_spin_semantics",
    "descriptor_shell_sizes",
    "projector_sha256",
    "reference_family",
    "response_backend",
}


def _as_checkpoint_metadata(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_checkpoint_metadata(item) for item in value]
    if isinstance(value, dict):
        return {
            _as_checkpoint_metadata(key): _as_checkpoint_metadata(item)
            for key, item in value.items()
        }
    raise TypeError(
        f"checkpoint metadata does not support {type(value).__name__} values"
    )


def normalize_force_contract_fingerprint(value) -> str:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.uint8 or value.shape != (32,):
            raise ValueError(
                "force contract fingerprint tensor must have dtype uint8 and shape (32,)"
            )
        value = bytes(value.detach().cpu().tolist())
    elif isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.shape != (32,):
            raise ValueError(
                "force contract fingerprint array must have dtype uint8 and shape (32,)"
            )
        value = value.tobytes()
    elif isinstance(value, (list, tuple)):
        try:
            value = bytes(value)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid force contract fingerprint bytes") from error
    if isinstance(value, bytes):
        if len(value) != 32:
            raise ValueError("force contract fingerprint must contain exactly 32 bytes")
        return value.hex()
    if not isinstance(value, str):
        raise TypeError("force contract fingerprint must be bytes or a hexadecimal string")
    fingerprint = value.lower()
    if fingerprint.startswith("sha256:"):
        fingerprint = fingerprint[7:]
    if len(fingerprint) != 64:
        raise ValueError("force contract fingerprint must contain 64 hexadecimal digits")
    try:
        bytes.fromhex(fingerprint)
    except ValueError as error:
        raise ValueError("force contract fingerprint must be hexadecimal") from error
    return fingerprint


def _validate_checkpoint_force_metadata(
    extra_info,
    *,
    expected_fingerprint=None,
    expected_contract=None,
):
    """Validate and return strict relaxed-force checkpoint metadata."""
    if not isinstance(extra_info, Mapping):
        raise ValueError("checkpoint extra_info must be a mapping")
    metadata = extra_info.get("force_training")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint is missing force_training metadata")
    if set(metadata) != FORCE_CHECKPOINT_METADATA_KEYS:
        missing = sorted(FORCE_CHECKPOINT_METADATA_KEYS - set(metadata))
        extra = sorted(set(metadata) - FORCE_CHECKPOINT_METADATA_KEYS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(
            "force-training checkpoint metadata fields are invalid: "
            + "; ".join(details)
        )
    if metadata.get("jacobian_semantics") != FORCE_JACOBIAN_SEMANTICS:
        raise ValueError(
            "force-training checkpoint must declare dq_dR_relaxed Jacobian semantics"
        )
    fingerprint = normalize_force_contract_fingerprint(
        metadata.get("compatibility_fingerprint")
    )
    if expected_fingerprint is not None:
        expected = normalize_force_contract_fingerprint(expected_fingerprint)
        if fingerprint != expected:
            raise ValueError(
                "force-training checkpoint contract fingerprint does not match the data"
            )
    if expected_contract is not None:
        from deepks.data.force_schema import validate_force_checkpoint_metadata

        validate_force_checkpoint_metadata(metadata, expected_contract)
    return dict(metadata)


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_loaded_model_force_contract(model, contract) -> None:
    try:
        dimensions = contract.dimensions
        projector_basis = contract.manifest["descriptor"]["projector_basis"]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(
            "expected_force_contract is not a validated force-data contract"
        ) from error
    expected_features = int(dimensions["n_feature"])
    if model.input_dim != expected_features:
        raise ValueError(
            "checkpoint model input dimension does not match the force-data contract"
        )
    if _metadata_signature(model._pbas) != _metadata_signature(
        load_basis(projector_basis)
    ):
        raise ValueError(
            "checkpoint model projector metadata does not match the force-data contract"
        )


def parse_actv_fn(code):
    if callable(code):
        return code
    assert type(code) is str
    lcode = code.lower()
    if lcode == 'sigmoid':
        return torch.sigmoid
    if lcode == 'tanh':
        return torch.tanh
    if lcode == 'relu':
        return torch.relu
    if lcode == 'softplus':
        return F.softplus
    if lcode == 'silu':
        return F.silu
    if lcode == 'gelu':
        return F.gelu
    if lcode == 'mygelu':
        return mygelu
    raise ValueError(f'{code} is not a valid activation function')


def make_embedder(type, shell_sec, **kwargs):
    ltype = type.lower()
    if ltype in ("trace", "sum"):
        EmbdCls = TraceEmbedding
    elif ltype in ("thermal", "softmax"):
        EmbdCls = ThermalEmbedding
    else:
        raise ValueError(f'{type} is not a valid embedding type')
    embedder = EmbdCls(shell_sec, **kwargs)
    return embedder


def mygelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def log_args(name):
    def decorator(func):
        def warpper(self, *args, **kwargs):
            args_dict = inspect.getcallargs(func, self, *args, **kwargs)
            del args_dict['self']
            setattr(self, name, args_dict)
            func(self, *args, **kwargs)
        return warpper
    return decorator


def make_shell_mask(shell_sec):
    lsize = len(shell_sec)
    msize = max(shell_sec)
    mask = torch.zeros(lsize, msize, dtype=bool)
    for l, m in enumerate(shell_sec):
        mask[l, :m] = 1
    return mask


def pad_lastdim(sequences, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    front_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])
    out_dims = front_dims + (len(sequences), max_len)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[..., i, :length] = tensor
    return out_tensor


def pad_masked(tensor, mask, padding_value=0):
    # equiv to pad_lastdim(tensor.split(shell_sec, dim=-1))
    assert tensor.shape[-1] == mask.sum()
    new_shape = tensor.shape[:-1] + mask.shape
    return tensor.new_full(new_shape, padding_value).masked_scatter_(mask, tensor) 


def unpad_lastdim(padded, length_list):
    # inverse of pad_lastdim
    return [padded[...,i,:length] for i, length in enumerate(length_list)]


def unpad_masked(padded, mask):
    # equiv to torch.cat(unpad_lastdim(padded, shell_sec), dim=-1)
    new_shape = padded.shape[:-mask.ndim] + (mask.sum(),)
    return torch.masked_select(padded, mask).reshape(new_shape)


def masked_softmax(input, mask, dim=-1):
    exps = torch.exp(input - input.max(dim=dim, keepdim=True)[0])
    mexps = exps * mask.to(exps)
    msums = mexps.sum(dim=dim, keepdim=True).clamp(1e-10)
    return mexps / msums


class DenseNet(nn.Module):
    
    def __init__(self, sizes, actv_fn=torch.relu, use_resnet=True, with_dt=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) 
                                     for in_f, out_f in zip(sizes, sizes[1:])])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
        if with_dt:
            self.dts = nn.ParameterList(
                [nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) 
                    for out_f in sizes[1:]])
        else:
            self.dts = None
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and layer.in_features == layer.out_features:
                if self.dts is not None:
                    tmp = tmp * self.dts[i]
                x = x + tmp
            else:
                x = tmp
        return x


class TraceEmbedding(nn.Module):

    def __init__(self, shell_sec):
        super().__init__()
        self.shell_sec = shell_sec
        self.ndesc = len(shell_sec)
    
    def forward(self, x):
        x_shells = x.split(self.shell_sec, dim=-1)
        tr_shells = [sx.sum(-1, keepdim=True) for sx in x_shells]
        return torch.cat(tr_shells, dim=-1)
    

class ThermalEmbedding(nn.Module):

    def __init__(self, shell_sec, embd_sizes=None, init_beta=5., 
                 momentum=None, max_memory=1000):
        super().__init__()
        self.shell_sec = shell_sec
        self.register_buffer("shell_mask", make_shell_mask(shell_sec), False)# shape: [l, m]
        if embd_sizes is None:
            embd_sizes = shell_sec
        if isinstance(embd_sizes, int):
            embd_sizes = [embd_sizes] * len(shell_sec)
        assert len(embd_sizes) == len(shell_sec)
        self.embd_sizes = embd_sizes
        self.register_buffer("embd_mask", make_shell_mask(embd_sizes), False)
        self.ndesc = sum(embd_sizes)
        self.beta = nn.Parameter( # shape: [l, p], padded
            pad_lastdim([torch.linspace(init_beta, -init_beta, ne) 
                            for ne in embd_sizes]))
        self.momentum = momentum
        self.max_memory = max_memory
        self.register_buffer('running_mean', torch.zeros(len(shell_sec)))
        self.register_buffer('running_var', torch.ones(len(shell_sec)))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_padded = pad_masked(x, self.shell_mask, 0.) # shape: [n, a, l, m]
        if self.training:
            self.update_running_stats(x_padded)
        nx_padded = ((x_padded - self.running_mean.unsqueeze(-1)) 
                    / (self.running_var.sqrt().unsqueeze(-1) + SCALE_EPS)
                    * self.shell_mask.to(x_padded))
        weight = masked_softmax(
            torch.einsum("...lm,lp->...lmp", nx_padded, -self.beta),
            self.shell_mask.unsqueeze(-1), dim=-2)
        desc_padded = torch.einsum("...m,...mp->...p", x_padded, weight)
        return unpad_masked(desc_padded, self.embd_mask)

    def update_running_stats(self, x_padded):
        self.num_batches_tracked += 1
        if self.momentum is None and self.num_batches_tracked > self.max_memory:
            return # stop update after 1000 batches, so the scaling becomes a fixed parameter
        exp_factor = 1. - 1. / float(self.num_batches_tracked)
        if self.momentum is not None:
            exp_factor = max(exp_factor, self.momentum)
        with torch.no_grad():
            fmask = self.shell_mask.to(x_padded)
            pad_portion = fmask.mean(-1)
            x_masked = x_padded * fmask # make sure padded part is zero
            reduced_dim = (*range(x_masked.ndim-2), -1)
            batch_mean = x_masked.mean(reduced_dim) / pad_portion
            batch_var = x_masked.var(reduced_dim) / pad_portion
            self.running_mean[:] = exp_factor * self.running_mean + (1-exp_factor) * batch_mean
            self.running_var[:] = exp_factor * self.running_var + (1-exp_factor) * batch_var
        
    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()


class CorrNet(nn.Module):

    @log_args('_init_args')
    def __init__(self, input_dim, hidden_sizes=(100,100,100), 
                 actv_fn='gelu', use_resnet=True, 
                 embedding=None, proj_basis=None, elem_table=None,
                 input_shift=0, input_scale=1, output_scale=1):
        super().__init__()
        actv_fn = parse_actv_fn(actv_fn)
        self.input_dim = input_dim
        # basis info
        self._pbas = load_basis(proj_basis)
        self._init_args["proj_basis"] = self._pbas
        self.shell_sec = None
        # elem const
        if isinstance(elem_table, str):
            elem_table = load_elem_table(elem_table)
            self._init_args["elem_table"] = elem_table
        self.elem_table = elem_table
        self.elem_dict = None if elem_table is None else dict(zip(*elem_table))
        # linear fitting
        self.linear = nn.Linear(input_dim, 1).double()
        # embedding net
        ndesc = input_dim
        self.embedder = None
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = {"type": embedding}
            assert isinstance(embedding, dict)
            raw_shell_sec = get_shell_sec(self._pbas)
            self.shell_sec = raw_shell_sec * (input_dim // sum(raw_shell_sec))
            assert sum(self.shell_sec) == input_dim
            self.embedder = make_embedder(**embedding, shell_sec=self.shell_sec).double()
            self.linear.requires_grad_(False) # make sure it is symmetric
            ndesc = self.embedder.ndesc
        # fitting net
        layer_sizes = [ndesc, *hidden_sizes, 1]
        self.densenet = DenseNet(layer_sizes, actv_fn, use_resnet).double()
        # scaling part
        self.input_shift = nn.Parameter(
            torch.tensor(input_shift, dtype=torch.float64).expand(input_dim).clone(), 
            requires_grad=False)
        self.input_scale = nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float64).expand(input_dim).clone(), 
            requires_grad=False)
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64), 
            requires_grad=False)
        self.energy_const = nn.Parameter(
            torch.tensor(0, dtype=torch.float64), 
            requires_grad=False)
        self._checkpoint_extra_info = {}
    
    def forward(self, x):
        # x: nframes x natom x nfeature
        x = (x - self.input_shift) / (self.input_scale + SCALE_EPS)
        l = self.linear(x)
        if self.embedder is not None:
            x = self.embedder(x)
        y = self.densenet(x)
        y = y / self.output_scale + l
        e = y.sum(-2) + self.energy_const
        return e
    
    def get_elem_const(self, elems):
        if self.elem_dict is None:
            return 0.
        return sum(self.elem_dict[ee] for ee in elems)

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data[:] = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data[:] = torch.tensor(scale, dtype=dtype)

    def set_prefitting(self, weight, bias, trainable=False):
        dtype = self.linear.weight.dtype
        self.linear.weight.data[:] = torch.tensor(weight, dtype=dtype).reshape(-1)
        self.linear.bias.data[:] = torch.tensor(bias, dtype=dtype).reshape(-1)
        self.linear.requires_grad_(trainable)

    def set_energy_const(self, const):
        dtype = self.energy_const.dtype
        self.energy_const.data = torch.tensor(const, dtype=dtype).reshape([])

    def save_dict(self, **extra_info):
        retained_extra_info = dict(self._checkpoint_extra_info)
        retained_extra_info.update(extra_info)
        dump_dict = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": self.state_dict(),
            "init_args": _as_checkpoint_metadata(self._init_args),
            "extra_info": _as_checkpoint_metadata(retained_extra_info),
        }
        return dump_dict

    def save(self, filename, **extra_info):
        torch.save(self.save_dict(**extra_info), filename)

    @staticmethod
    def load_dict(
        checkpoint,
        strict=False,
        *,
        require_force_metadata=False,
        expected_force_contract_fingerprint=None,
        expected_force_contract=None,
    ):
        if not isinstance(checkpoint, Mapping):
            raise TypeError("CorrNet checkpoint must be a mapping")
        format_version = checkpoint.get("format_version")
        if format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported CorrNet checkpoint format: {format_version!r}"
            )
        extra_info = checkpoint.get("extra_info", {})
        if not isinstance(extra_info, Mapping):
            raise ValueError("CorrNet checkpoint extra_info must be a mapping")
        has_force_metadata = "force_training" in extra_info
        if (
            require_force_metadata
            or expected_force_contract_fingerprint is not None
            or expected_force_contract is not None
        ):
            _validate_checkpoint_force_metadata(
                extra_info,
                expected_fingerprint=expected_force_contract_fingerprint,
                expected_contract=expected_force_contract,
            )
            has_force_metadata = True
        elif has_force_metadata:
            _validate_checkpoint_force_metadata(extra_info)
        init_args = dict(checkpoint["init_args"])
        if "layer_sizes" in init_args:
            layers = init_args.pop("layer_sizes")
            init_args["input_dim"] = layers[0]
            init_args["hidden_sizes"] = layers[1:-1]
        model = CorrNet(**init_args)
        model.load_state_dict(
            checkpoint['state_dict'],
            strict=True if has_force_metadata else strict,
        )
        if expected_force_contract is not None:
            _validate_loaded_model_force_contract(model, expected_force_contract)
        model._checkpoint_extra_info = _as_checkpoint_metadata(dict(extra_info))
        return model

    @staticmethod
    def load(
        filename,
        strict=False,
        *,
        require_force_metadata=False,
        expected_force_contract_fingerprint=None,
        expected_force_contract=None,
    ):
        checkpoint = torch.load(
            filename,
            map_location="cpu",
            weights_only=True,
        )
        return CorrNet.load_dict(
            checkpoint,
            strict=strict,
            require_force_metadata=require_force_metadata,
            expected_force_contract_fingerprint=expected_force_contract_fingerprint,
            expected_force_contract=expected_force_contract,
        )
