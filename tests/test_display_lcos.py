import torch
import math

from pado.display import LCOSLUT, lcos_encode_phase


def test_lut_gray_to_phase_endpoints() -> None:
    lut = LCOSLUT(phase_lut=torch.tensor([0.0, 1.0, 2.0]))
    g0 = torch.tensor(0.0)
    g1 = torch.tensor(1.0)
    assert float(lut.gray_to_phase(g0)) == 0.0
    assert float(lut.gray_to_phase(g1)) == 2.0


def test_lut_phase_to_gray_roundtrip_monotonic() -> None:
    phase_lut = torch.linspace(0.0, 2.0, 33)
    lut = LCOSLUT(phase_lut=phase_lut)
    g = torch.rand(4, 1, 8, 8)
    p = lut.gray_to_phase(g)
    g2 = lut.phase_to_gray(p)
    assert torch.max(torch.abs(g2 - g)).item() < 1e-5


def test_ste_quantization_preserves_grad() -> None:
    phase_lut = torch.linspace(-3.14, 3.14, 257)
    lut = LCOSLUT(phase_lut=phase_lut)

    phase = torch.zeros(1, 1, 4, 4, requires_grad=True)
    out = lcos_encode_phase(phase, lut, wvl=None, bits=8, ste=True, wrap=False)
    loss = out["gray"].sum()
    loss.backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all().item()


def test_positive_phase_lut_wraps_target_into_positive_domain() -> None:
    lut = LCOSLUT(phase_lut=torch.linspace(0.0, 2 * math.pi, 257))
    phase = torch.tensor([[[[-math.pi / 2]]]])

    out = lcos_encode_phase(phase, lut, bits=None, ste=True, wrap=True)

    assert torch.allclose(
        out["phase_realized"],
        torch.tensor([[[[1.5 * math.pi]]]]),
        atol=1e-5,
    )


def test_bipolar_phase_lut_keeps_bipolar_wrapped_phase() -> None:
    lut = LCOSLUT(phase_lut=torch.linspace(-math.pi, math.pi, 257))
    phase = torch.tensor([[[[-math.pi / 2]]]])

    out = lcos_encode_phase(phase, lut, bits=None, ste=True, wrap=True)

    assert torch.allclose(
        out["phase_realized"],
        torch.tensor([[[[-math.pi / 2]]]]),
        atol=1e-5,
    )
