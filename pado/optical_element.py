########################################################
# The MIT License (MIT)
#
# PADO (Pytorch Automatic Differentiable Optics)
# Copyright (c) 2023 by POSTECH Computer Graphics Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Contact:
# Lead Developer: Dong-Ha Shin (0218sdh@gmail.com)
# Corresponding Author: Seung-Hwan Baek (shwbaek@postech.ac.kr)
#
########################################################

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F

from .math import wrap_phase
from .math import nm, um, mm, cm, m
from .light import Light
from .material import Material


def _interpolate_spatial(
    tensor: torch.Tensor,
    *,
    scale_factor: float,
    mode: str,
) -> torch.Tensor:
    kwargs = {"scale_factor": scale_factor, "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)


class OpticalElement:
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, wvl: float, 
                 field_change: Optional[torch.Tensor] = None, device: str = 'cpu', 
                 name: str = "not defined", polar: str = 'non') -> None:
        """Base class for optical elements that modify incident light wavefront.

        The wavefront modification is stored as amplitude and phase tensors.
        Note that the number of channels is one for wavefront modulation.

        Args:
            dim (tuple): Dimensions (B, 1, R, C) for batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength of light in meters
            field_change (torch.Tensor, optional): Wavefront modification tensor [B, C, H, W]
            device (str): Device to store wavefront ('cpu', 'cuda:0', etc.)
            name (str): Name identifier for this optical element
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.field_change.shape
            torch.Size([1, 1, 100, 100])
        """
        self.name = name
        self.dim = dim
        self.pitch = pitch
        self.device = device
        if field_change is None:
            self.field_change = torch.ones(dim, dtype=torch.cfloat, device=device)
        else:
            self.field_change = field_change
        self.wvl = wvl
        self.polar = polar

    def forward(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Propagate incident light through the optical element.

        Args:
            light (Light): Input light field
            interp_mode (str): Interpolation method for resizing ('bilinear', 'nearest')

        Returns:
            Light: Light field after interaction with optical element

        Examples:
            >>> element = OpticalElement(dim=(1, 1, 64, 64), pitch=2e-6)
            >>> light = Light(dim=(1, 1, 64, 64), pitch=2e-6)
            >>> output = element.forward(light)
        """
        if light.pitch > self.pitch:
            light.resize(self.pitch, interp_mode)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch, interp_mode)
            self.set_pitch(light.pitch)

        if self.polar=='non':
            return self.forward_non_polar(light, interp_mode)
        elif self.polar=='polar':
            x = self.forward_non_polar(light.get_lightX(), interp_mode)
            y = self.forward_non_polar(light.get_lightY(), interp_mode)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            raise NotImplementedError('Polar is not set.')

    def forward_non_polar(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Propagate non-polarized light through the optical element.

        Handles resolution matching between light and optical element by resizing and padding
        as needed. Applies the optical element's field modulation to the input light.

        Args:
            light (Light): Input light field to propagate through the element
            interp_mode (str): Interpolation method for resizing ('bilinear', 'nearest')

        Returns:
            Light: Modified light field after interaction with optical element

        Raises:
            ValueError: If wavelengths of light and element don't match
        """
        if light.wvl != self.wvl:
            raise ValueError(f'Wavelength mismatch: light wavelength {light.wvl} != element wavelength {self.wvl}')

        # make sure that light and optical element have the same resolution, i.e. pixel count, by padding the smaller one
        r1 = np.abs((light.dim[2] - self.dim[2])//2)
        r2 = np.abs(light.dim[2] - self.dim[2]) - r1
        pad_width = (r1, r2, 0, 0)
        if light.dim[2] > self.dim[2]:
            self.pad(pad_width)
        elif light.dim[2] < self.dim[2]:
            light.pad(pad_width)

        c1 = np.abs((light.dim[3] - self.dim[3])//2)
        c2 = np.abs(light.dim[3] - self.dim[3]) - c1
        pad_width = (0, 0, c1, c2)
        if light.dim[3] > self.dim[3]:
            self.pad(pad_width)
        elif light.dim[3] < self.dim[3]:
            light.pad(pad_width)

        light.set_field(light.field*self.field_change)

        return light

    def get_amplitude_change(self) -> torch.Tensor:
        """Return amplitude change of the wavefront.

        Returns:
            torch.Tensor: Amplitude change of the wavefront

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> amp = element.get_amplitude_change()
        """
        return self.field_change.abs()

    def get_device(self) -> str:
        """Returns the device on which tensors are stored.

        Returns:
            str: The device identifier (e.g., 'cpu', 'cuda:0').
        """
        return self.device

    def get_field_change(self) -> torch.Tensor:
        """Returns the field_change tensor.

        Returns:
            torch.Tensor: The field_change tensor representing amplitude and phase changes.
        """
        return self.field_change

    def get_name(self) -> str:
        """Returns the name of the optical element.

        Returns:
            str: The name identifier.
        """
        return self.name

    def get_phase_change(self) -> torch.Tensor:
        """Return phase change of the wavefront.

        Returns:
            torch.Tensor: Phase change of the wavefront

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> phase = element.get_phase_change()
        """
        return self.field_change.angle()

    def get_pitch(self) -> float:
        """Returns the pixel pitch.

        Returns:
            float: The pixel pitch in meters.
        """
        return self.pitch

    def get_polar(self) -> str:
        """Returns the polarization mode.

        Returns:
            str: The polarization mode ('non' for scalar, 'polar' for vector).
        """
        return self.polar

    def get_wvl(self) -> float:
        """Returns the wavelength.

        Returns:
            float: The wavelength in meters.
        """
        return self.wvl

    def pad(self, pad_width: Tuple[int, int, int, int], padval: float = 0) -> None:
        """Pad the wavefront change with constant value.

        Args:
            pad_width (tuple): Padding width following torch.nn.functional.pad format
            padval (float): Value to pad with, only 0 supported currently

        Raises:
            NotImplementedError: If padval is not 0

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.pad((10,10,10,10))  # Add 10 pixels padding on all sides
        """
        if padval == 0:
            # Create a new padded tensor instead of modifying in-place
            padded_field_change = torch.nn.functional.pad(self.field_change, pad_width)
            self.field_change = padded_field_change
        else:
            raise NotImplementedError('only zero padding supported')

        # Create a new dim tuple instead of modifying in-place
        new_dim = list(self.dim)
        new_dim[2], new_dim[3] = new_dim[2]+pad_width[2]+pad_width[3], new_dim[3]+pad_width[0]+pad_width[1]
        self.dim = tuple(new_dim)

    def resize(self, target_pitch: float, interp_mode: str = 'nearest') -> None:
        """Resize the wavefront change by changing the pixel pitch.

        Args:
            target_pitch (float): New pixel pitch to use
            interp_mode (str): Interpolation method used in torch.nn.functional.interpolate
                - 'bilinear': Bilinear interpolation
                - 'nearest': Nearest neighbor interpolation

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.resize(1e-6)  # Resize to 1μm pitch
        """
        scale_factor = self.pitch / target_pitch
        resized_real = _interpolate_spatial(self.field_change.real, scale_factor=scale_factor, mode=interp_mode)
        resized_imag = _interpolate_spatial(self.field_change.imag, scale_factor=scale_factor, mode=interp_mode)
        self.field_change = torch.complex(resized_real, resized_imag)
        
        # Create a new dim tuple instead of modifying in-place
        new_dim = list(self.dim)
        new_dim[2], new_dim[3] = self.field_change.shape[2], self.field_change.shape[3]
        self.dim = tuple(new_dim)
        self.set_pitch(target_pitch)

    def set_amplitude_change(self, amplitude: torch.Tensor, c: Optional[int] = None) -> None:
        """Set amplitude change for specific or all channels.

        Args:
            amplitude (torch.Tensor): Amplitude change in polar representation
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> amp = torch.ones((1,1,100,100))
            >>> element.set_amplitude_change(amp)
        """
        if c is not None:
            phase = self.field_change[:, c, ...].angle()
            # Create a new field change tensor instead of modifying in-place
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change
        else:
            phase = self.field_change.angle()
            # Create a completely new tensor
            new_field_change = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change

    def set_field_change(self, field_change: torch.Tensor, c: Optional[int] = None) -> None:
        """Set field change for specific or all channels.

        Args:
            field_change (torch.Tensor): Field change in complex tensor
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> field = torch.ones((1,1,100,100), dtype=torch.cfloat)
            >>> element.set_field_change(field)
        """
        if c is not None:
            # Create a clone to avoid in-place modification
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = field_change
            self.field_change = new_field_change
        else:
            # Create a new tensor for all channels
            new_field_change = self.field_change.clone()
            for chan in range(self.dim[1]):
                new_field_change[:, chan, ...] = field_change
            self.field_change = new_field_change

    def set_name(self, name: str) -> None:
        """Sets the name of the optical element.

        Args:
            name (str): The name identifier.
        """
        self.name = name

    def set_phase_change(self, phase: torch.Tensor, c: Optional[int] = None) -> None:
        """Set phase change for specific or all channels.

        Args:
            phase (torch.Tensor): Phase change in polar representation
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> phase = torch.zeros((1,1,100,100))
            >>> element.set_phase_change(phase)
        """
        if c is not None:
            amplitude = self.field_change[:, c, ...].abs()
            # Create a clone to avoid in-place modification
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change
        else:
            # Create a new tensor for all channels
            new_field_change = self.field_change.clone()
            for chan in range(self.dim[1]):
                amplitude = self.field_change[:, chan, ...].abs()
                new_field_change[:, chan, ...] = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change

    def set_pitch(self, pitch: float) -> None:
        """Set the pixel pitch of the complex tensor.

        Args:
            pitch (float): Pixel pitch in meters

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.set_pitch(1e-6)  # Set 1μm pitch
        """
        if pitch <= 0:
            raise ValueError(f"Pitch must be positive, got {pitch}")
        self.pitch = pitch

    def set_polar(self, polar: str) -> None:
        """Set polarization mode for the optical element.

        Args:
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.set_polar('polar')  # Set vector field mode
        """
        self.polar = polar

    def set_wvl(self, wvl: float) -> None:
        """Sets the wavelength.

        Args:
            wvl (float): The wavelength in meters.
        """
        self.wvl = wvl

    def shape(self) -> Tuple[int, int, int, int]:
        """Return shape of light-wavefront modulation.

        The number of channels is one for wavefront modulation.

        Returns:
            tuple: Dimensions (B, 1, R, C) for batch size, channels, rows, columns

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.shape()
            (1, 1, 100, 100)
        """
        return self.dim

    def visualize(self, b: int = 0, c: Optional[int] = None) -> None:
        """Visualize the wavefront modulation of the optical element.

        Displays amplitude and phase changes of the optical element's wavefront modulation.
        Creates subplots showing amplitude change and phase change for specified channels.

        Args:
            b (int, optional): Batch index to visualize. Defaults to 0.
            c (int, optional): Channel index to visualize. If None, visualizes all channels.
                Defaults to None.

        Examples:
            >>> lens = RefractiveLens((1,1,512,512), 2e-6, 0.1, 633e-9, 'cpu')
            >>> lens.visualize()  # Visualize first batch, all channels
            >>> lens.visualize(b=0, c=0)  # Visualize first batch, first channel
        """
        channels = [c] if c is not None else range(self.dim[1])

        for chan in channels:
            plt.figure(figsize=(13,6))
            plt.subplot(121)
            plt.imshow(self.get_amplitude_change().data.cpu()[b,chan,...].squeeze(), cmap='inferno', vmin=0, vmax=1)
            plt.title('amplitude change')
            plt.colorbar()
            
            plt.subplot(122)
            plt.imshow(self.get_phase_change().data.cpu()[b,chan,...].squeeze(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
            plt.title('phase change')
            plt.colorbar()
            
            wvl_text = f'{self.wvl[chan]/nm:.2f}[nm]' if isinstance(self.wvl, list) else f'{self.wvl/nm:.2f}[nm]'
            plt.suptitle(
                f'{self.name}, '
                f'({self.dim[2]},{self.dim[3]}), '
                f'pitch:{self.pitch/um:.2f}[um], '
                f'wvl:{wvl_text}, '
                f'device:{self.device}'
            )

class RefractiveLens(OpticalElement):
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, focal_length: float, 
                 wvl: Union[float, List[float]], device: str, polar: str = 'non', 
                 designated_wvl: Optional[float] = None) -> None:
        """Create a thin refractive lens optical element.

        Simulates a thin refractive lens that modifies the phase of incident light
        based on its focal length and wavelength.

        Args:
            dim (tuple): Shape of the lens field (B, Ch, R, C) where:
                B: Batch size
                Ch: Number of channels
                R: Number of rows
                C: Number of columns
            pitch (float): Pixel pitch in meters
            focal_length (float): Focal length of the lens in meters
            wvl (float or list): Wavelength(s) of light in meters. Can be single value or list for multi-channel
            device (str): Device to store the lens field ('cpu', 'cuda:0', etc.)
            polar (str, optional): Polarization mode. Defaults to 'non'
            designated_wvl (float, optional): Override wavelength for all channels. Defaults to None

        Examples:
            >>> # Create single channel lens
            >>> lens = RefractiveLens((1,1,512,512), 2e-6, 0.1, 633e-9, 'cpu')
            
            >>> # Create multi-channel lens with different wavelengths
            >>> lens = RefractiveLens((1,3,512,512), 2e-6, 0.1, [633e-9,532e-9,450e-9], 'cuda:0')
        """
        super().__init__(dim, pitch, wvl, None, device, name="refractive_lens", polar=polar)

        self.focal_length: Optional[float] = None

        if focal_length is None:
            raise ValueError("focal_length cannot be None")
            
        self.set_focal_length(focal_length)

        if dim[1] == 1:
            phase = self.compute_phase(self.wvl, shift_x=0, shift_y=0)
            # Create unit amplitude field with exact 1.0 amplitude
            amplitude = torch.ones(phase.shape, dtype=torch.float64, device=self.device)
            field_change = amplitude * torch.exp(1j * phase)
            self.set_field_change(field_change, c=0)
        else:
            if designated_wvl is not None:
                for i in range(dim[1]):
                    phase = self.compute_phase(designated_wvl, shift_x=0, shift_y=0)
                    # Create unit amplitude field with exact 1.0 amplitude
                    amplitude = torch.ones(phase.shape, dtype=torch.float64, device=self.device)
                    field_change = amplitude * torch.exp(1j * phase)
                    self.set_field_change(field_change, c=i)
            else:
                for i in range(dim[1]):
                    phase = self.compute_phase(self.wvl[i], shift_x=0, shift_y=0)
                    # Create unit amplitude field with exact 1.0 amplitude
                    amplitude = torch.ones(phase.shape, dtype=torch.float64, device=self.device)
                    field_change = amplitude * torch.exp(1j * phase)
                    self.set_field_change(field_change, c=i)

    def set_focal_length(self, focal_length: float) -> None:
        """Set the focal length of the lens.

        Args:
            focal_length (float): New focal length in meters

        Examples:
            >>> lens.set_focal_length(0.2)  # Set 20cm focal length
        """
        self.focal_length = focal_length

    def compute_phase(self, wvl: float, shift_x: float = 0, shift_y: float = 0) -> torch.Tensor:
        """Compute the phase modulation for the lens.

        Calculates the phase change introduced by the lens based on its focal length,
        wavelength and any lateral shifts.

        Args:
            wvl (float): Wavelength of light in meters
            shift_x (float, optional): Horizontal displacement of lens center in meters. Defaults to 0
            shift_y (float, optional): Vertical displacement of lens center in meters. Defaults to 0

        Returns:
            torch.Tensor: Phase modulation pattern of the lens

        Examples:
            >>> phase = lens.compute_phase(633e-9)  # Centered lens
            >>> phase = lens.compute_phase(633e-9, shift_x=10e-6)  # Shifted lens
        """
        x = np.arange(-self.dim[3]/2, self.dim[3]/2) * self.pitch
        y = np.arange(-self.dim[2]/2, self.dim[2]/2) * self.pitch
        xx, yy = np.meshgrid(x, y, indexing='xy')

        theta_change = torch.tensor((-2*np.pi / wvl)*((xx-shift_x)**2 + (yy-shift_y)**2), device=self.device) / (2*self.focal_length)
        theta_change = (theta_change + np.pi) % (np.pi * 2) - np.pi
        theta_change = torch.unsqueeze(torch.unsqueeze(theta_change, axis=0), axis=0)
        
        return theta_change


class CosineSquaredLens(OpticalElement):
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, focal_length: float, 
                 wvl: float, device: str, polar: str = 'non') -> None:
        """Lens with cosine squared phase distribution.

        Creates a lens with phase distribution of form [1+cos(k*r^2)]/2.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) - batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            device (str): Device to store wavefront ('cpu', 'cuda:0', ...)
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> # Create basic cosine squared lens
            >>> lens = CosineSquaredLens((1,1,1024,1024), 2e-6, 0.1, 633e-9, 'cpu')
        """
        super().__init__(dim, pitch, wvl, None, device, name="cosine_squared_lens", polar=polar)
        
        self.focal_length: float = focal_length
        self.compute_and_set_phase_change()

    def compute_and_set_phase_change(self) -> None:
        """Compute and set the phase change induced by the lens.

        Calculates and applies phase change in range [0, π] to the lens.

        Examples:
            >>> lens.compute_and_set_phase_change()
        """
        k = 20 * np.pi / self.wvl  # Wave number
        
        x = np.arange(-self.dim[3]/2, self.dim[3]/2) * self.pitch
        y = np.arange(-self.dim[2]/2, self.dim[2]/2) * self.pitch
        xx, yy = np.meshgrid(x, y, indexing='xy')
        
        xx = torch.tensor(xx, device=self.device)
        yy = torch.tensor(yy, device=self.device)
        
        r_squared = xx**2 + yy**2  # Radius squared from the center
        
        # Calculate phase change based on pi*[1+cos(k*r^2)]/2 to adjust the range to [0, pi]
        phase_change = np.pi * (1 + torch.cos(k * r_squared)) / 2
        phase_change = torch.unsqueeze(torch.unsqueeze(phase_change, axis=0), axis=0)
        
        for i in range(self.dim[1]):  # Assuming potential multiple wavelengths or batch dimension
            self.set_phase_change(phase_change, c=i)


def height2phase(height: float, wvl: float, RI: float, wrap: bool = True) -> torch.Tensor:
    """Convert material height to corresponding phase shift.

    Calculates phase shift from material height using wavelength and refractive index.

    Args:
        height (float): Height of material in meters
        wvl (float): Wavelength of light in meters
        RI (float): Refractive index of material at given wavelength
        wrap (bool): If True, wraps phase to [0,2π] range

    Returns:
        torch.Tensor: Phase change induced by material height

    Examples:
        >>> height = 500e-9  # 500nm height
        >>> phase = height2phase(height, 633e-9, 1.5)
    """
    dRI = RI - 1
    wv_n = 2. * np.pi / wvl
    phi = wv_n * dRI * height
    if wrap:
        phi = wrap_phase(phi, stay_positive=True)
    return phi

def phase2height(phase_u: torch.Tensor, wvl: float, RI: float, minh: float = 0) -> torch.Tensor:
    """Convert phase change to material height.

    Note that phase to height mapping is not one-to-one.
    There exists an integer phase wrapping factor
    ``height = wvl/(RI-1) * (phase_u + i*2π)``, where ``i`` is an integer.
    This function uses minimum height minh to constrain the conversion.
    Minimal height is chosen such that height is always >= minh.

    Args:
        phase_u (torch.Tensor): Phase change of light
        wvl (float): Wavelength of light in meters
        RI (float): Refractive index of material at given wavelength
        minh (float): Minimum height constraint in meters

    Returns:
        torch.Tensor: Material height that induces the phase change

    Examples:
        >>> phase = torch.ones((1,1,1024,1024)) * np.pi
        >>> height = phase2height(phase, 633e-9, 1.5, minh=100e-9)  # 100nm min height
    """
    dRI = RI - 1
    if minh is not None:
        i = torch.ceil(((dRI/wvl)*minh - phase_u)/(2*np.pi))
    else:
        i = 0
    height = wvl * (phase_u + 2*np.pi*i) / dRI
    return height


class DOE(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, material: 'Material', wvl: float, device: str, height: Optional[torch.Tensor] = None, phase_change: Optional[torch.Tensor] = None, polar: str = 'non'):
        """Diffractive optical element (DOE) that modifies incident light wavefront.

        The wavefront modification is determined by the material height profile.
        Supports both height and phase change specifications.

        Args:
            dim (tuple): Dimensions (B, 1, R, C) for batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            material (Material): Material properties of the DOE
            wvl (float): Wavelength of light in meters
            device (str): Device to store wavefront ('cpu', 'cuda:0', etc.)
            height (torch.Tensor, optional): Height profile in meters
            phase_change (torch.Tensor, optional): Phase change profile
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> # Create DOE with specified height profile
            >>> height = torch.ones((1,1,100,100)) * 500e-9  # 500nm height
            >>> doe = DOE(height.shape, 2e-6, material, 500e-9, 'cpu', height=height)
            
            >>> # Create DOE with specified phase profile
            >>> phase = torch.ones((1,1,100,100)) * np.pi  # π phase
            >>> doe = DOE(phase.shape, 2e-6, material, 500e-9, 'cpu', phase_change=phase)
        """
        super().__init__(dim=dim, pitch=pitch, wvl=wvl, device=device, name="doe", polar=polar)

        self.material: 'Material' = material
        self.height: Optional[torch.Tensor] = None

        # initial DOE is tranparent and induces 0 phase delay
        super().set_field_change(torch.ones(dim,device=device)*torch.exp(1*torch.zeros(dim,device=device)))

        if (height is None) and (phase_change is not None):
            self.set_phase_change(phase_change, sync_height=True)
        elif (height is not None) and (phase_change is None):
            self.set_height(height, sync_phase=True)
        elif (height is None) and (phase_change is None):
            phase = torch.zeros(dim, device=device)
            self.set_phase_change(phase, sync_height=True)

    def visualize(self, b: int = 0, c: int = 0) -> None:
        """Visualize the DOE wavefront modulation.

        Displays amplitude change, phase change and height profile.

        Args:
            b (int): Batch index to visualize, defaults to 0
            c (int): Channel index to visualize, defaults to 0

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.visualize()  # Shows modulation plots
            >>> doe.visualize(b=1, c=0)  # Shows plots for batch index 1
        """
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.imshow(self.get_amplitude_change().data.cpu()[b,c,...].squeeze(), 
                   cmap='inferno', vmin=0, vmax=1)
        plt.title('amplitude change')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(self.get_phase_change().data.cpu()[b,c,...].squeeze(), 
                   cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.title('phase change')
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow(self.get_height().data.cpu()[b,c,...].squeeze()*1e6, 
                   cmap='hot')
        plt.title('height [um]')
        plt.colorbar()
        
        plt.suptitle(
            f'{self.name}, '
            f'({self.dim[2]},{self.dim[3]}), '
            f'pitch:{self.pitch/1e-6:.2f}[um], '
            f'wvl:{self.wvl/1e-9:.2f}[nm], '
            f'device:{self.device}'
        )
        plt.show()

    def set_diffraction_grating_1d(self, slit_width: float, minh: float, maxh: float) -> None:
        """Set the wavefront modulation as a 1D diffraction grating.

        Create alternating height regions to form a binary phase grating.

        Args:
            slit_width (float): Width of each slit in meters
            minh (float): Minimum height in meters
            maxh (float): Maximum height in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_diffraction_grating_1d(10e-6, 0, 500e-9)  # 10μm slits
            >>> doe.visualize()  # Shows 1D grating pattern
        """
        slit_width_px = int(np.round(slit_width / self.pitch))
        if slit_width_px < 1:
            raise ValueError(f"slit_width={slit_width} is too small for pitch={self.pitch} (slit_width_px={slit_width_px})")

        slit_space_px = slit_width_px

        dg = np.zeros((self.dim[2], self.dim[3]))
        slit_num_r = self.dim[2] // (2 * slit_width_px)
        slit_num_c = self.dim[3] // (2 * slit_width_px)

        # Create a copy to avoid modifying in-place
        dg_copy = dg.copy()
        dg_copy[:] = minh

        for i in range(int(slit_num_c)):
            minc = int((slit_width_px + slit_space_px) * i)
            maxc = int(minc + slit_width_px)

            dg_copy[:, minc:maxc] = maxh
        # dg_copy is a height map in meters. Use height->phase conversion via material RI.
        height = torch.tensor(dg_copy.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        self.set_height(height, sync_phase=True)

    def set_diffraction_grating_2d(self, slit_width: float, minh: float, maxh: float) -> None:
        """Set the wavefront modulation as a 2D diffraction grating.

        Create a checkerboard pattern of alternating height regions.

        Args:
            slit_width (float): Width of each slit in meters
            minh (float): Minimum height in meters
            maxh (float): Maximum height in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_diffraction_grating_2d(10e-6, 0, 500e-9)  # 10μm slits
            >>> doe.visualize()  # Shows 2D grating pattern
        """
        slit_width_px = int(np.round(slit_width / self.pitch))
        if slit_width_px < 1:
            raise ValueError(f"slit_width={slit_width} is too small for pitch={self.pitch} (slit_width_px={slit_width_px})")

        slit_space_px = slit_width_px

        dg = np.zeros((self.dim[2], self.dim[3]))
        slit_num_r = self.dim[2] // (2 * slit_width_px)
        slit_num_c = self.dim[3] // (2 * slit_width_px)

        # Create a copy to avoid modifying in-place
        dg_copy = dg.copy()
        dg_copy[:] = minh

        for i in range(int(slit_num_r)):
            for j in range(int(slit_num_c)):
                minc = int((slit_width_px + slit_space_px) * j)
                maxc = int(minc + slit_width_px)
                minr = int((slit_width_px + slit_space_px) * i)
                maxr = int(minr + slit_width_px)

                dg_copy[minr:maxr, minc:maxc] = maxh

        # dg_copy is a height map in meters. Use height->phase conversion via material RI.
        height = torch.tensor(dg_copy.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        self.set_height(height, sync_phase=True)

    def set_Fresnel_lens(self, focal_length: float, wvl: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set the wavefront modulation as a Fresnel lens.

        Create a phase profile that focus light to a point.

        Args:
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            shift_x (float): Horizontal shift in meters. Defaults to 0
            shift_y (float): Vertical shift in meters. Defaults to 0

        Examples:
            >>> doe = DOE((1,1,1024,1024), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_Fresnel_lens(0.1, 500e-9)  # f=10cm lens
            >>> doe.set_Fresnel_lens(0.1, 500e-9, shift_x=50e-6)  # Shifted lens
        """
        x = np.arange(
            -self.dim[3] * self.pitch / 2,
            self.dim[3] * self.pitch / 2,
            self.pitch
        )
        x = x[:self.dim[3]]
        
        y = np.arange(
            -self.dim[2] * self.pitch / 2,
            self.dim[2] * self.pitch / 2,
            self.pitch
        )
        y = y[:self.dim[2]]
        
        xx, yy = np.meshgrid(x, y)
        xx = torch.tensor(xx, device=self.device)
        yy = torch.tensor(yy, device=self.device)

        phase_u = (-2 * np.pi / wvl) * (
            torch.sqrt(
                (xx - shift_x)**2 + 
                (yy - shift_y)**2 + 
                focal_length**2
            ) - focal_length
        )
        
        phase_w = wrap_phase(phase_u)
        phase_w = phase_w.unsqueeze(0).unsqueeze(0)

        self.set_phase_change(phase_w, sync_height=True)
    
    def set_Fresnel_zone_plate_lens(self, focal_length: float, wvl: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set binary Fresnel zone plate pattern.

        Creates alternating opaque and transparent zones that focus light.

        Args:
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            shift_x (float): Horizontal shift in meters. Defaults to 0
            shift_y (float): Vertical shift in meters. Defaults to 0

        Examples:
            >>> doe = DOE((1,1,1024,1024), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_Fresnel_zone_plate_lens(0.1, 500e-9)  # f=10cm lens
            >>> doe.set_Fresnel_zone_plate_lens(0.1, 500e-9, shift_x=50e-6)  # Shifted lens
        """
        x = np.arange(-self.dim[3]/2, self.dim[3]/2) * self.pitch
        y = np.arange(-self.dim[2]/2, self.dim[2]/2) * self.pitch
        xx, yy = np.meshgrid(x, y, indexing='xy')

        # Calculate the radial distance from the center
        r_squared = (xx - shift_x)**2 + (yy - shift_y)**2

        # Original phase calculation for a thin lens
        original_phase = (-2 * np.pi / wvl) * r_squared / (2 * focal_length)

        # Fresnel zone plate phase calculation
        # Map phase to 0 or pi based on the sign of the cosine of the original phase
        fresnel_phase = np.pi * (np.cos(original_phase) >= 0).astype(np.float32)

        fresnel_phase = torch.tensor(fresnel_phase, device=self.device)
        fresnel_phase = torch.unsqueeze(torch.unsqueeze(fresnel_phase, axis=0), axis=0)
        
        self.set_phase_change(fresnel_phase, sync_height=True)
    
    def change_wvl(self, wvl: float) -> None:
        """Change the wavelength and update phase change.

        Args:
            wvl (float): New wavelength in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.change_wvl(633e-9)  # Change to 633nm wavelength
        """
        height = self.get_height()
        # Store the new wavelength
        self.wvl = wvl
        # Calculate the phase change for the new wavelength
        phase = height2phase(height, self.wvl, self.material.get_RI(self.wvl))
        # Create a new field change tensor
        new_field_change = torch.exp(phase*1j)
        self.set_field_change(new_field_change, sync_height=False)

    def set_height(self, height: torch.Tensor, sync_phase: bool = True) -> None:
        """Set the height map of the DOE.

        Args:
            height (torch.Tensor): Height map in meters
            sync_phase (bool): If True, syncs phase profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> height = torch.ones((1,1,100,100)) * 500e-9
            >>> doe.set_height(height, sync_phase=True)
        """
        # Store a copy of the height tensor to avoid potential shared memory with input
        self.height = height.clone() if height is not None else None
        if sync_phase:  
            self.sync_phase_with_height()
            
    def sync_height_with_phase(self) -> None:
        """Synchronize height profile with current phase profile.

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_phase_change(phase, sync_height=False)
            >>> doe.sync_height_with_phase()  # Update height to match phase
        """
        height = phase2height(self.get_phase_change(), self.wvl, self.material.get_RI(self.wvl))
        self.set_height(height, sync_phase=False)

    def sync_phase_with_height(self) -> None:
        """Synchronize phase profile with current height profile.

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_height(height, sync_phase=False)
            >>> doe.sync_phase_with_height()  # Update phase to match height
        """
        phase = height2phase(self.get_height(), self.wvl, self.material.get_RI(self.wvl))
        self.set_phase_change(phase, sync_height=False)

    def resize(self, target_pitch: float) -> None:
        """Resize DOE with a new pixel pitch.

        Resize field from which DOE height is recomputed.

        Args:
            target_pitch (float): New pixel pitch in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.resize(1e-6)  # Change pitch to 1μm
        """
        super().resize(target_pitch)  # this changes the field change 
        self.sync_height_with_phase()

    def get_height(self) -> torch.Tensor:
        """Return the height map of the DOE.

        Returns:
            torch.Tensor: Height map in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> height = doe.get_height()  # Get current height profile
        """
        return self.height
    
    def set_phase_change(self, phase_change: torch.Tensor, sync_height: bool = True) -> None:
        """Set phase change induced by the DOE.

        Args:
            phase_change (torch.Tensor): Phase change profile
            sync_height (bool): If True, syncs height profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> phase = torch.ones((1,1,100,100)) * np.pi
            >>> doe.set_phase_change(phase, sync_height=True)
        """
        super().set_phase_change(phase_change)
        if sync_height:
            self.sync_height_with_phase()

    def set_field_change(self, field_change: torch.Tensor, sync_height: bool = True) -> None:
        """Change the field change of the DOE.

        Args:
            field_change (torch.Tensor): Complex field change tensor
            sync_height (bool): If True, syncs height profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> field = torch.exp(1j * torch.ones((1,1,100,100)))
            >>> doe.set_field_change(field, sync_height=True)
        """
        super().set_field_change(field_change)
        if sync_height:
            self.sync_height_with_phase()

    def set_height(self, height: torch.Tensor, sync_phase: bool = True) -> None:
        """Set the height map of the DOE.

        Args:
            height (torch.Tensor): Height map in meters
            sync_phase (bool): If True, syncs phase profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> height = torch.ones((1,1,100,100)) * 500e-9
            >>> doe.set_height(height, sync_phase=True)
        """
        self.height = height.clone() if height is not None else None
        if sync_phase:
            self.sync_phase_with_height()


class SLM(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, wvl: float, device: str, polar: str = 'non'):
        """Spatial Light Modulator (SLM) optical element.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)
            polar (str): Polarization mode ('non' or 'polar')

        Examples:
            >>> slm = SLM(dim=(1,1,1024,1024), pitch=6.4e-6, wvl=633e-9, device='cuda:0')
        """
        super().__init__(dim, pitch, wvl, device=device, name="SLM", polar=polar)

    def set_lens(self, focal_length: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set phase profile to implement a thin lens.

        Args:
            focal_length (float): Focal length in meters
            shift_x (float): Lateral shift in x direction in meters
            shift_y (float): Lateral shift in y direction in meters

        Examples:
            >>> slm.set_lens(focal_length=0.5, shift_x=100e-6)  # 500mm focal length, 100μm x-shift
        """
        x = np.arange(-self.dim[3]*self.pitch/2, self.dim[3]*self.pitch/2, self.pitch)
        y = np.arange(-self.dim[2]*self.pitch/2, self.dim[2]*self.pitch/2, self.pitch)
        xx,yy = np.meshgrid(x,y)

        phase_u = (2*np.pi / self.wvl)*((xx-shift_x)**2 + (yy-shift_y)**2) / (2*focal_length)
        phase_u = torch.tensor(phase_u.astype(np.float32), device=self.device).unsqueeze(0).unsqueeze(0)
        phase_w = wrap_phase(phase_u, stay_positive=False)
        self.set_phase_change(phase_w)

    def set_amplitude_change(self, amplitude: torch.Tensor, wvl: Optional[float] = None) -> None:
        """Set amplitude modulation profile of the SLM.

        Args:
            amplitude (torch.Tensor): Amplitude modulation profile [B, 1, R, C]
            wvl (float): Operating wavelength in meters

        Examples:
            >>> amp = torch.ones((1,1,1024,1024)) * 0.8  # 80% transmission
            >>> slm.set_amplitude_change(amp, wvl=633e-9)
        """
        if wvl is not None:
            self.wvl = wvl
        super().set_amplitude_change(amplitude)

    def set_phase_change(self, phase_change: torch.Tensor, wvl: Optional[float] = None) -> None:
        """Set phase modulation profile of the SLM.

        Args:
            phase_change (torch.Tensor): Phase modulation profile [B, 1, R, C] in radians
            wvl (float): Operating wavelength in meters

        Examples:
            >>> phase = torch.ones((1,1,1024,1024)) * np.pi  # π phase shift
            >>> slm.set_phase_change(phase, wvl=633e-9)
        """
        if wvl is not None:
            self.wvl = wvl
        super().set_phase_change(phase_change)
        
        
class PolarizedSLM(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, wvl: float, device: str):
        """SLM which can control phase & amplitude of each polarization component.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)

        Examples:
            >>> slm = PolarizedSLM(dim=(1,1,1024,1024), pitch=6.4e-6, wvl=633e-9, device='cuda:0')
        """
        super().__init__(dim, pitch, wvl, device=device, name="Metasurface", polar='polar')
        self.amplitude_change = torch.ones((dim[0], 1, dim[2], dim[3], 2), device=self.device)
        self.phase_change = torch.zeros((dim[0], 1, dim[2], dim[3], 2), device=self.device)

    def get_amplitude_change(self) -> torch.Tensor:
        return self.amplitude_change

    def get_phase_change(self) -> torch.Tensor:
        return self.phase_change

    def set_amplitude_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for both polarization components.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C, 2] in polar representation
            wvl (float): Wavelength in meters

        Examples:
            >>> amp = torch.ones((1,1,1024,1024,2)) * 0.8  # 80% transmission for both polarizations
            >>> slm.set_amplitude_change(amp, wvl=633e-9)
        """
        if amplitude.shape != self.amplitude_change.shape:
            raise ValueError(f"Expected amplitude shape {self.amplitude_change.shape}, got {amplitude.shape}")
        self.wvl = wvl
        self.amplitude_change = amplitude.clone()

    def set_phase_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for both polarization components.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C, 2] in polar representation
            wvl (float): Wavelength in meters

        Examples:
            >>> phase = torch.ones((1,1,1024,1024,2)) * np.pi  # π phase shift for both polarizations
            >>> slm.set_phase_change(phase, wvl=633e-9)
        """
        if phase_change.shape != self.phase_change.shape:
            raise ValueError(f"Expected phase shape {self.phase_change.shape}, got {phase_change.shape}")
        self.wvl = wvl
        self.phase_change = phase_change.clone()

    def set_amplitudeX_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for X polarization component.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C] for X component
            wvl (float): Wavelength in meters

        Examples:
            >>> ampX = torch.ones((1,1,1024,1024)) * 0.8  # 80% transmission for X polarization
            >>> slm.set_amplitudeX_change(ampX, wvl=633e-9)
        """
        self.wvl = wvl
        amp = self.get_amplitude_change()
        # Create new tensor instead of modifying in-place
        new_amp = amp.clone()
        new_amp[:,:,:,:,0] = amplitude
        self.amplitude_change = new_amp

    def set_amplitudeY_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for Y polarization component.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C] for Y component
            wvl (float): Wavelength in meters

        Examples:
            >>> ampY = torch.ones((1,1,1024,1024)) * 0.6  # 60% transmission for Y polarization
            >>> slm.set_amplitudeY_change(ampY, wvl=633e-9)
        """
        self.wvl = wvl
        amp = self.get_amplitude_change()
        # Create new tensor instead of modifying in-place
        new_amp = amp.clone()
        new_amp[:,:,:,:,1] = amplitude
        self.amplitude_change = new_amp

    def set_phaseX_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for X polarization component.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C] for X component
            wvl (float): Wavelength in meters

        Examples:
            >>> phaseX = torch.ones((1,1,1024,1024)) * np.pi/2  # π/2 phase shift for X polarization
            >>> slm.set_phaseX_change(phaseX, wvl=633e-9)
        """
        self.wvl = wvl
        phase = self.get_phase_change()
        # Create new tensor instead of modifying in-place
        new_phase = phase.clone()
        new_phase[:,:,:,:,0] = phase_change
        self.phase_change = new_phase

    def set_phaseY_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for Y polarization component.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C] for Y component
            wvl (float): Wavelength in meters

        Examples:
            >>> phaseY = torch.ones((1,1,1024,1024)) * np.pi  # π phase shift for Y polarization
            >>> slm.set_phaseY_change(phaseY, wvl=633e-9)
        """
        self.wvl = wvl
        phase = self.get_phase_change()
        # Create new tensor instead of modifying in-place
        new_phase = phase.clone()
        new_phase[:,:,:,:,1] = phase_change
        self.phase_change = new_phase

    def get_phase_changeX(self) -> torch.Tensor:
        """Return phase change for X polarization component.

        Returns:
            torch.Tensor: Phase change [B, 1, R, C] for X component

        Examples:
            >>> phaseX = slm.get_phase_changeX()  # Get X polarization phase profile
        """
        return self.get_phase_change()[:,:,:,:,0]

    def get_phase_changeY(self) -> torch.Tensor:
        """Return phase change for Y polarization component.

        Returns:
            torch.Tensor: Phase change [B, 1, R, C] for Y component

        Examples:
            >>> phaseY = slm.get_phase_changeY()  # Get Y polarization phase profile
        """
        return self.get_phase_change()[:,:,:,:,1]

    def get_amplitude_changeX(self) -> torch.Tensor:
        """Return amplitude change for X polarization component.

        Returns:
            torch.Tensor: Amplitude change [B, 1, R, C] for X component

        Examples:
            >>> ampX = slm.get_amplitude_changeX()  # Get X polarization amplitude profile
        """
        return self.get_amplitude_change()[:,:,:,:,0]

    def get_amplitude_changeY(self) -> torch.Tensor:
        """Return amplitude change for Y polarization component.

        Returns:
            torch.Tensor: Amplitude change [B, 1, R, C] for Y component

        Examples:
            >>> ampY = slm.get_amplitude_changeY()  # Get Y polarization amplitude profile
        """
        return self.get_amplitude_change()[:,:,:,:,1]
        
    def forward(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Apply polarization-dependent modulation to input light.

        Args:
            light (Light): Input light field
            interp_mode (str): Interpolation mode for resizing ('nearest', 'bilinear', etc.)

        Returns:
            Light: Modulated light field

        Examples:
            >>> modulated_light = slm.forward(input_light)  # Apply polarization modulation
            >>> modulated_light = slm.forward(input_light, interp_mode='bilinear')  # Use bilinear interpolation
        """
        if light.wvl != self.wvl:
            raise ValueError(f'Wavelength mismatch: light wavelength {light.wvl} != element wavelength {self.wvl}')
        
        if light.pitch > self.pitch:
            light.resize(self.pitch, interp_mode)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch, interp_mode)
            self.set_pitch(light.pitch)
            
        r1 = np.abs((light.dim[2] - self.dim[2])//2)
        r2 = np.abs(light.dim[2] - self.dim[2]) - r1
        pad_width = (r1, r2, 0, 0)
        if light.dim[2] > self.dim[2]:
            self.pad(pad_width)
        elif light.dim[2] < self.dim[2]:
            light.pad(pad_width)

        c1 = np.abs((light.dim[3] - self.dim[3])//2)
        c2 = np.abs(light.dim[3] - self.dim[3]) - c1
        pad_width = (0, 0, c1, c2)
        if light.dim[3] > self.dim[3]:
            self.pad(pad_width)
        elif light.dim[3] < self.dim[3]:
            light.pad(pad_width)
        
        # Ensure compatible shapes for broadcasting
        light_phase = light.get_phase()
        phase_change = self.get_phase_change()
        
        # Check if shapes are compatible and reshape if needed
        if light_phase.shape != phase_change.shape:
            # Ensure light_phase has the same shape as phase_change for proper broadcasting
            if light_phase.dim() == 4 and phase_change.dim() == 5:
                # Add polarization dimension if missing
                light_phase = light_phase.unsqueeze(-1)
                if light_phase.shape[-1] == 1:
                    # Duplicate the phase for both polarizations
                    light_phase = light_phase.expand(-1, -1, -1, -1, 2)
        
        # Apply phase modulation with proper shape handling
        phase = (light_phase + phase_change + np.pi) % (np.pi*2) - np.pi
        
        # Set the modulated phase and amplitude for each polarization component - non-modifying operations
        light.set_phaseX(phase[..., 0])
        light.set_phaseY(phase[..., 1])
        light.set_amplitudeX(light.get_amplitudeX() * self.get_amplitude_change()[..., 0])
        light.set_amplitudeY(light.get_amplitudeY() * self.get_amplitude_change()[..., 1])
        
        return light

    def pad(self, pad_width: tuple, padval: int = 0) -> None:
        """Pad amplitude and phase changes with constant value.

        Args:
            pad_width (tuple): Padding dimensions (left, right, top, bottom)
            padval (int): Padding value (only 0 supported)

        Examples:
            >>> slm.pad((16,16,16,16))  # Add 16 pixels padding on all sides
        """
        if padval == 0:
            # Create new padded tensors instead of modifying in-place
            padded_amplitude = torch.nn.functional.pad(
                self.get_amplitude_change(), 
                (0,0,0,0,pad_width[2],pad_width[3],pad_width[0],pad_width[1])
            )
            padded_phase = torch.nn.functional.pad(
                self.get_phase_change(), 
                (0,0,0,0,pad_width[2],pad_width[3],pad_width[0],pad_width[1])
            )
            self.amplitude_change = padded_amplitude
            self.phase_change = padded_phase
        else:
            raise NotImplementedError('only zero padding supported')

        # Create a new dim tuple instead of modifying in-place
        new_dim = list(self.dim)
        new_dim[2] += pad_width[0] + pad_width[1]
        new_dim[3] += pad_width[2] + pad_width[3]
        self.dim = tuple(new_dim)
        
    def visualize(self, b: int = 0) -> None:
        """Visualize amplitude and phase modulation for both polarizations.

        Args:
            b (int): Batch index to visualize, default 0

        Examples:
            >>> slm.visualize()  # Visualize first batch
            >>> slm.visualize(b=1)  # Visualize second batch
        """
        plt.figure(figsize=(13,8))
        
        plt.subplot(221)
        plt.imshow(self.get_amplitude_changeX().data.cpu()[b,...].squeeze(), cmap='inferno')
        plt.title('amplitude change X')
        plt.colorbar()
        
        plt.subplot(222)
        plt.imshow(self.get_phase_changeX().data.cpu()[b,...].squeeze(), cmap='hsv')
        plt.title('phase change X')
        plt.colorbar()
        
        plt.subplot(223)
        plt.imshow(self.get_amplitude_changeY().data.cpu()[b,...].squeeze(), cmap='inferno')
        plt.title('amplitude change Y')
        plt.colorbar()
        
        plt.subplot(224)
        plt.imshow(self.get_phase_changeY().data.cpu()[b,...].squeeze(), cmap='hsv')
        plt.title('phase change Y')
        plt.colorbar()
        
        plt.suptitle(
            f'{self.name}, '
            f'({self.dim[2]},{self.dim[3]}), '
            f'pitch:{self.pitch/1e-6:.2f}[um], '
            f'wvl:{self.wvl/1e-9:.2f}[nm], '
            f'device:{self.device}'
        )
        plt.show()


class Aperture(OpticalElement):
    """Aperture optical element for amplitude modulation.
    
    Implement square or circular aperture that modulate light amplitude.
    Support both polarized and non-polarized light.
    """

    def __init__(self, dim: tuple, pitch: float, aperture_diameter: float, aperture_shape: str, wvl: float, device: str = 'cpu', polar: str = 'non'):
        """Create aperture optical element instance.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            aperture_diameter (float): Diameter of aperture in meters
            aperture_shape (str): Shape of aperture ('square' or 'circle')
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)
            polar (str): Polarization mode ('non', 'x', 'y', 'xy')

        Examples:
            >>> aperture = Aperture(dim=(1,1,1024,1024), pitch=6.4e-6, 
            ...                    aperture_diameter=1e-3, aperture_shape='circle',
            ...                    wvl=633e-9)
        """
        super().__init__(dim, pitch, wvl, device=device, name="aperture", polar=polar)

        self.aperture_diameter = aperture_diameter
        self.aperture_shape = aperture_shape
        self.amplitude_change = torch.zeros((self.dim[2], self.dim[3]), device=device)
        if self.aperture_shape == 'square':
            self.set_square()
        elif self.aperture_shape == 'circle':
            self.set_circle()
        else:
            return NotImplementedError

    def set_square(self) -> None:
        """Set square aperture amplitude modulation.

        Create square aperture mask centered on optical axis.

        Examples:
            >>> aperture.set_square()
        """
        self.aperture_shape = 'square'

        [x, y] = np.mgrid[-self.dim[2]//2:self.dim[2]//2, -self.dim[3]//2:self.dim[3]//2].astype(np.float32)
        r = self.pitch * np.asarray([abs(x), abs(y)]).max(axis=0)
        r = np.expand_dims(np.expand_dims(r, axis=0), axis=0)

        max_val = self.aperture_diameter / 2
        amp = (r <= max_val).astype(np.float32)
        amp_copy = amp.copy()  # Make a copy to avoid modifying the array in-place
        amp_copy[amp_copy == 0] = 1e-20
        self.set_field_change(torch.tensor(amp_copy, device=self.device))

    def set_circle(self, cx: float = 0, cy: float = 0, dia: float = None) -> None:
        """Set circular aperture amplitude modulation.

        Create circular aperture mask with optional offset and diameter.

        Args:
            cx (float): Center x-offset in pixels
            cy (float): Center y-offset in pixels  
            dia (float, optional): Circle diameter in meters

        Examples:
            >>> aperture.set_circle()  # Centered circle
            >>> aperture.set_circle(cx=10, cy=-10, dia=2e-3)  # Offset circle
        """
        [x, y] = np.mgrid[-self.dim[2]//2:self.dim[2]//2, -self.dim[3]//2:self.dim[3]//2].astype(np.float32)
        r2 = (x-cx) ** 2 + (y-cy) ** 2
        r2_copy = r2.copy()
        r2_copy[r2_copy < 0] = 1e-20
        r = self.pitch * np.sqrt(r2_copy)
        r = np.expand_dims(np.expand_dims(r, axis=0), axis=0)
        
        if dia is not None:
            self.aperture_diameter = dia
        self.aperture_shape = 'circle'
        max_val = self.aperture_diameter / 2
        amp = (r <= max_val).astype(np.float32)
        amp_copy = amp.copy()  # Make a copy to avoid modifying the array in-place
        amp_copy[amp_copy == 0] = 1e-20
        self.set_field_change(torch.tensor(amp_copy, device=self.device))


def quantize(x: Union[torch.Tensor, np.ndarray], levels: int, vmin: float = None, vmax: float = None, include_vmax: bool = True) -> Union[torch.Tensor, np.ndarray]:
    """Quantize floating point array.

    Discretize input array into specified number of levels.

    Args:
        x (torch.Tensor or np.ndarray): Input array to quantize
        levels (int): Number of quantization levels
        vmin (float, optional): Minimum value for quantization
        vmax (float, optional): Maximum value for quantization
        include_vmax (bool): Whether to include max value in quantization
            True: Quantize with spacing of 1/levels
            False: Quantize with spacing of 1/(levels-1)

    Returns:
        torch.Tensor or np.ndarray: Quantized array

    Examples:
        >>> x = torch.randn(100)
        >>> x_quant = quantize(x, levels=8)
        >>> x_quant = quantize(x, levels=16, vmin=-1, vmax=1)
    """
    if include_vmax is False:
        if levels == 0:
            return x

        if vmin is None:
            vmin = x.min()
        if vmax is None:
            vmax = x.max()

        normalized = (x - vmin) / (vmax - vmin + 1e-16)
        if isinstance(x, np.ndarray):
            levelized = np.floor(normalized * levels) / (levels - 1)
        elif isinstance(x, torch.Tensor):    
            levelized = (normalized * levels).floor() / (levels - 1)
        result = levelized * (vmax - vmin) + vmin
        
        # Create a copy to ensure no in-place operations
        if isinstance(x, np.ndarray):
            result_copy = result.copy()
            result_copy[result_copy < vmin] = vmin
            result_copy[result_copy > vmax] = vmax
            return result_copy
        else:  # torch.Tensor
            # For tensors, we use clamp which returns a new tensor
            return torch.clamp(result, min=vmin, max=vmax)
    
    elif include_vmax is True:
        space = (x.max()-x.min())/levels
        vmin = x.min()
        vmax = vmin + space*(levels-1)
        if isinstance(x, np.ndarray):
            result = (np.floor((x-vmin)/space))*space + vmin
            # Create a copy to ensure no in-place operations
            result_copy = result.copy()
            result_copy[result_copy < vmin] = vmin
            result_copy[result_copy > vmax] = vmax
            return result_copy
        elif isinstance(x, torch.Tensor):    
            result = (((x-vmin)/space).floor())*space + vmin
            # For tensors, we use clamp which returns a new tensor
            return torch.clamp(result, min=vmin, max=vmax)
