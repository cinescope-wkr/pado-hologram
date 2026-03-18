from pathlib import Path

import torch

from pado.light import Light, PolarizedLight
from pado.math import calculate_ssim
from pado.optical_element import OpticalElement, PolarizedSLM, SLM


def test_light_pad_keeps_dim_in_sync_with_tensor_shape() -> None:
    light = Light((1, 1, 4, 5), pitch=1.0, wvl=1.0, device="cpu")
    light.pad((1, 2, 3, 4))

    assert light.field.shape == torch.Size((1, 1, 11, 8))
    assert light.dim == (1, 1, 11, 8)


def test_optical_element_pad_keeps_dim_in_sync_with_tensor_shape() -> None:
    element = OpticalElement((1, 1, 4, 5), pitch=1.0, wvl=1.0, device="cpu")
    element.pad((1, 2, 3, 4))

    assert element.field_change.shape == torch.Size((1, 1, 11, 8))
    assert element.dim == (1, 1, 11, 8)


def test_light_load_image_supports_random_phase_for_single_batch() -> None:
    image_path = Path(__file__).resolve().parents[1] / "example" / "asset" / "A.png"

    light = Light((2, 1, 4, 4), pitch=1.0, wvl=1.0, device="cpu")
    light.load_image(str(image_path), random_phase=True, batch_idx=0)

    phase = light.get_phase()
    assert torch.isfinite(phase).all().item()
    assert torch.allclose(phase[1], torch.zeros_like(phase[1]))


def test_light_magnify_resizes_complex_field_without_crashing() -> None:
    light = Light((1, 2, 4, 4), pitch=1.0, wvl=[1.0, 2.0], device="cpu")
    light.magnify(2.0, c=0)

    assert light.field.shape == torch.Size((1, 2, 8, 8))
    assert light.dim == (1, 2, 8, 8)


def test_polarized_light_clone_and_shape_updates_work() -> None:
    field_x = torch.ones((1, 1, 4, 4), dtype=torch.cfloat)
    field_y = torch.zeros((1, 1, 4, 4), dtype=torch.cfloat)
    light = PolarizedLight((1, 1, 4, 4), pitch=1.0, wvl=1.0, fieldX=field_x, fieldY=field_y, device="cpu")

    clone = light.clone()
    assert torch.equal(clone.get_fieldX(), field_x)
    assert torch.equal(clone.get_fieldY(), field_y)

    light.crop((1, 1, 1, 1))
    assert light.dim == (1, 1, 2, 2)

    light.magnify(2.0)
    assert light.dim == (1, 1, 4, 4)


def test_slm_set_lens_uses_current_wavelength() -> None:
    slm = SLM((1, 1, 4, 4), pitch=1.0, wvl=1.0, device="cpu")
    slm.set_lens(1.0)

    assert slm.get_phase_change().shape == torch.Size((1, 1, 4, 4))


def test_calculate_ssim_supports_multichannel_inputs() -> None:
    image = torch.rand(1, 3, 32, 32)

    ssim = calculate_ssim(image, image.clone())
    assert float(ssim) > 0.999


def test_polarized_slm_tracks_polarization_specific_state() -> None:
    slm = PolarizedSLM((1, 1, 4, 4), pitch=1.0, wvl=1.0, device="cpu")
    amplitude = torch.full((1, 1, 4, 4, 2), 0.5)
    phase = torch.zeros((1, 1, 4, 4, 2))

    slm.set_amplitude_change(amplitude, wvl=1.0)
    slm.set_phase_change(phase, wvl=1.0)

    assert slm.get_amplitude_change().shape == torch.Size((1, 1, 4, 4, 2))
    assert slm.get_phase_change().shape == torch.Size((1, 1, 4, 4, 2))
