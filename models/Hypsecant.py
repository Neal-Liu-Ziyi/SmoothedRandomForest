import numpy as np
import torch


class HyperbolicSecant:

    name = 'hypsec'
    full_name = 'Hyperbolic Secant'

    def __init__(self, loc=0, scale=1):
        """
        Initialize the Hyperbolic Secant distribution with location and scale parameters.
        
        Parameters:
        loc (float): The location parameter (mean shift).
        scale (float): The scale parameter (spread of the distribution).
        """
        scale_arr = np.asarray(scale)
        if np.any(scale_arr <= 0):
            raise ValueError("Scale parameter must be positive.")
        self.loc = loc
        self.scale = scale

    @staticmethod
    def _compute_cdf(x, loc, scale):
        """
        Static method for computing CDF.
        """
        scale_arr = np.asarray(scale)
        if np.any(scale_arr <= 0):
            raise ValueError("Scale parameter must be positive.")

        # Convert x to a numpy array for vectorized computation
        x_arr = np.asarray(x)
        z = (x_arr - loc) / scale
        
        # Prepare an output array of the same shape as z
        cdf = np.empty_like(z, dtype=np.float64)
        
        # For z > 0, avoid overflow by using the transformed expression
        mask = z > 0
        cdf[mask] = 1 - (2 / np.pi) * np.arctan(np.exp(-z[mask]))
        # For z <= 0, compute directly
        cdf[~mask] = (2 / np.pi) * np.arctan(np.exp(z[~mask]))
        
        # If the input was a scalar, return a scalar
        if np.isscalar(x) or x_arr.ndim == 0:
            return cdf.item()
        return cdf

    @classmethod
    def cdf(cls, x, loc=0, scale=1):
        """
        Class method for computing CDF.
        
        Parameters:
        x (float or array-like): The point(s) at which to compute the CDF.
        loc (float, optional): The location parameter.
        scale (float or array-like, optional): The scale parameter.
        
        Returns:
        float or np.ndarray: CDF values at x.
        """
        return cls._compute_cdf(x, loc, scale)

    @staticmethod
    def torch_cdf(x, loc=0, scale=1):
        """
        PyTorch-compatible CDF computation with automatic differentiation support.
        
        Parameters:
        x (torch.Tensor): The point(s) at which to compute the CDF.
        loc (torch.Tensor or float): The location parameter.
        scale (torch.Tensor or float): The scale parameter.
        
        Returns:
        torch.Tensor: CDF values at x, supporting gradients.
        """
        # Ensure inputs are tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        
        # Parameter validation - clamp scale to reasonable range
        scale = torch.clamp(torch.abs(scale) + 1e-8, min=1e-6, max=1e6)
        
        # Standardize: z = (x - loc) / scale
        z = (x - loc) / scale
        
        # Clamp z to prevent overflow in exp()
        # exp(20) ≈ 4.85e8, exp(-20) ≈ 2.06e-9
        z = torch.clamp(z, min=-20.0, max=20.0)
        
        # Get pi with correct dtype and device
        pi = torch.tensor(torch.pi, dtype=z.dtype, device=z.device)
        
        # Numerically stable CDF computation
        # For z > 0: CDF = 1 - (2/π) * arctan(exp(-z))
        # For z <= 0: CDF = (2/π) * arctan(exp(z))
        
        # Compute both versions safely
        exp_pos = torch.exp(-torch.clamp(z, min=0))  # exp(-z) for z > 0
        exp_neg = torch.exp(torch.clamp(z, max=0))   # exp(z) for z <= 0
        
        cdf_positive = 1.0 - (2.0 / pi) * torch.atan(exp_pos)
        cdf_negative = (2.0 / pi) * torch.atan(exp_neg)
        
        # Use torch.where to maintain differentiability
        cdf = torch.where(z > 0, cdf_positive, cdf_negative)
        
        # Final safety check - clamp to valid probability range
        cdf = torch.clamp(cdf, min=1e-12, max=1.0 - 1e-12)
        
        return cdf

    @staticmethod
    def torch_pdf(x, loc=0, scale=1):
        """
        PyTorch-compatible PDF computation with automatic differentiation support.
        
        Parameters:
        x (torch.Tensor): The point(s) at which to compute the PDF.
        loc (torch.Tensor or float): The location parameter.
        scale (torch.Tensor or float): The scale parameter.
        
        Returns:
        torch.Tensor: PDF values at x, supporting gradients.
        """
        # Ensure inputs are tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        
        # Parameter validation - clamp scale to reasonable range
        scale = torch.clamp(torch.abs(scale) + 1e-8, min=1e-6, max=1e6)
        
        # Standardize: z = (x - loc) / scale
        z = (x - loc) / scale
        
        # Clamp z to prevent overflow in cosh
        z = torch.clamp(z, min=-20.0, max=20.0)
        
        # Get pi with correct dtype and device
        pi = torch.tensor(torch.pi, dtype=z.dtype, device=z.device)
        
        # PDF = sech(πz/2) / (2*scale) = 1/cosh(pi*z/2) / (2*scale)
        # Use clamp to prevent division by very large cosh values
        cosh_val = torch.cosh(pi * z / 2.0)
        cosh_val = torch.clamp(cosh_val, min=1.0, max=1e10)
        
        pdf = (1.0 / cosh_val) / (2.0 * scale)
        
        # Clamp PDF to reasonable range
        pdf = torch.clamp(pdf, min=1e-12, max=1e12)
        
        return pdf

    def _cdf(self, x):
        """
        Instance method for computing CDF using instance parameters.
        """
        return self._compute_cdf(x, self.loc, self.scale)
    
    def __str__(self):
        return f'HyperbolicSecant(loc={self.loc}, scale={self.scale})'
    
    def __name__(self):
        return self.name
    
    def __abbrev__(self):
        return self.abbrev