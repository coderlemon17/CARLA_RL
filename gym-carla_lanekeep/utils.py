from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tianshou.utils.net.common import MLP
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

SIGMA_MIN = -20
SIGMA_MAX = 2

class AttackedActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
        noise: str = 'none'
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.mu = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded
        self.noise = noise


    def my_forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        obs = torch.tensor(obs, requires_grad=True, device=self.device, dtype=torch.float32)
        # obs 4, l2, epsilon < 3e-2:
        noise_epsilon = 3e-1


        if self.noise == 'pgd':
            n = 50
            pgd_noise = noise_epsilon / n

            _obs = obs
            with torch.no_grad():
                (_mu, _sigma), state = self.my_forward(obs.detach().clone(), state, info)
            
            _noise = torch.randn_like(obs)
            _obs = _obs +  pgd_noise * _noise / (torch.norm(_noise) + 1e-10)
            for _ in range(n):
                _obs = _obs.detach().clone().requires_grad_()
                if _obs.grad is not None:
                    _obs.grad.zero_()
                # with torch.enable_grad():
                (mu, sigma), state = self.my_forward(_obs, state, info)
                kl_loss = kl_divergence(Normal(_mu, _sigma), Normal(torch.nan_to_num(mu), sigma)).sum()
                kl_loss.backward()

                _obs = _obs + pgd_noise * _obs.grad / (torch.norm(_obs.grad) + 1e-10)
            obs = obs + noise_epsilon * (_obs.detach().clone() - obs) / (torch.norm(_obs.detach().clone() - obs) + 1e-10)
        elif self.noise == 'random':
            noise = torch.randn_like(obs)
            obs = obs + noise_epsilon * noise / torch.norm(noise)
        elif self.noise == 'none':
            pass
        else:
            raise Exception(f'Unknown attack noise type {self.noise}')

        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state