"""Confirm the imported rasterizer is the DCSH spherical-harmonics build."""


def require_dcsh_rasterizer():
    from diff_gaussian_rasterization import _C

    variant_fn = getattr(_C, "sh_variant", None)
    got = variant_fn() if callable(variant_fn) else None
    if got != "dcsh":
        location = getattr(_C, "__file__", repr(_C))
        raise RuntimeError(
            "Training and rendering need the DCSH spherical-harmonics rasterizer "
            "built from submodules/diff-gaussian-rasterization-new. "
            f"The imported extension is {location} and reported variant {got!r}. "
            "Reinstall that directory. The empty submodule "
            "submodules/diff-gaussian-rasterization is vanilla 3DGS and will not load."
        )
    return _C
