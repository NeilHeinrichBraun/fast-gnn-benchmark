from collections.abc import Iterable
from typing import Any

import torch

MUON_EXCLUDED_NAME_PARTS = ("embedding_layer", "betas")

def split_muon_parameters(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:

    muon_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []

    for name, parameter in named_parameters:

        is_matrix = parameter.ndim == 2 and min(parameter.shape) > 1
        is_excluded = any(part in name for part in MUON_EXCLUDED_NAME_PARTS)

        if is_matrix and not is_excluded:
            muon_params.append(parameter)
        else:
            other_params.append(parameter)

    return muon_params, other_params


class MuonWithAuxAdamW(torch.optim.Optimizer):

    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        aux_params: list[torch.nn.Parameter],
        muon_kwargs: dict[str, Any],
        adamw_kwargs: dict[str, Any],
    ) -> None:
        if not muon_params:
            raise ValueError("Aucun parametre 2D a confier a Muon")

        param_groups: list[dict[str, Any]] = [{"params": muon_params}]
        if aux_params:
            param_groups.append({"params": aux_params})

        super().__init__(param_groups, defaults={})

        self.muon = torch.optim.Muon([self.param_groups[0]], **muon_kwargs)
        self.adamw = torch.optim.AdamW([self.param_groups[1]], **adamw_kwargs) if aux_params else None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict["adamw"] is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
