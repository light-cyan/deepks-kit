"""Self-consistent DeePKS methods."""

from .method import RDeePKS, UDeePKS


def DeePKS(mol, model, xc="HF", **kwargs):
    """Build the restricted or unrestricted method for a molecule."""
    method_class = RDeePKS if mol.spin == 0 else UDeePKS
    return method_class(mol, model, xc=xc, **kwargs)


__all__ = ["DeePKS", "RDeePKS", "UDeePKS"]
