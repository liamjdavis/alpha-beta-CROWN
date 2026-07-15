#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
AlphaValueData: A batch of Alpha **values**, mainly used in BaB interations.
AlphaFullInfoData: A batch of Alpha **values and related auxilaries**, used during .build() stage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from state.traits_mixin import DictLikeMixIn

if TYPE_CHECKING:
    from beta_CROWN_solver import LiRPANet

AlphaDataStartingNodeScope = Literal["part", "all"]
"""
* part: For each layer, starting_nodes=alpha_start_nodes (final node + --optimized_interm)
* all: For each layer, starting_nodes=all that present in the network
"""


@dataclass
class AlphaValueData(DictLikeMixIn):
    """
    A batch of Alpha **values**, mainly used in BaB interations.

    Format
    {activation_layer -> {
                final_layer -> tensor(2,1,batch,fixed_number),
                ... -> ...
            }
        ...
    }
    """

    _data: dict

    @staticmethod
    def from_net_part(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network with part scope (alpha_start_nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        net.alpha_drop_unused()
        alpha_start_nodes = net.alpha_start_nodes
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            raw_alpha_by_layer[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                if spec_name in alpha_start_nodes:
                    raw_alpha_by_layer[m.name][spec_name] = alpha.detach()
                    if move:
                        m.alpha[spec_name] = ValueError(
                            "Alpha has been popped and is no longer available "
                            "before assigning it again."
                        )

        return AlphaValueData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net_full(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network with full scope (all nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            raw_alpha_by_layer[m.name] = {}
            for spec_name, alpha in m.alpha.items():
                raw_alpha_by_layer[m.name][spec_name] = alpha.detach()
                if move:
                    m.alpha[spec_name] = ValueError(
                        "Alpha has been popped and is no longer available "
                        "before assigning it again."
                    )

        return AlphaValueData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net(
        net: "LiRPANet",
        *,
        starting_node_scope: Literal["part", "all"],
        move: bool = False,
    ) -> "AlphaValueData":
        """
        Extract alpha values from the network.
        Args:
            net: The network to extract from.
            starting_node_scope: The scope of starting nodes to extract. Additionally, if "part", call net.alpha_drop_unused() first.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaValueData instance containing the extracted alpha values
        """
        if starting_node_scope == "part":
            return AlphaValueData.from_net_part(net, move=move)
        elif starting_node_scope == "all":
            return AlphaValueData.from_net_full(net, move=move)
        else:
            raise ValueError(f"Unknown starting_node_scope: {starting_node_scope}")

    @staticmethod
    def from_domain_dict(d: dict) -> "AlphaValueData":
        """Create AlphaValueData from a domain dictionary from pick_out."""
        assert "alphas" in d
        return AlphaValueData(_data=d["alphas"])

    def attach_to_net(self, net: "LiRPANet"):
        """
        Attach all stored alpha values to the network.

        Args:
            net: The network to attach to.
        """

        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            if m.name in self._data:
                for spec_name, alpha in self._data[m.name].items():
                    m.alpha[spec_name] = alpha.detach().requires_grad_(True)

    def clone_tensor(self) -> "AlphaValueData":
        """Clone all tensors in the data structure."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha.clone() for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")

        return new_dict

    def mask_batch_dim_inplace(self, batch_mask) -> "AlphaValueData":
        """Inplace mask the batch dimension."""
        for _, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                for spec_name, alpha in layer_alpha.items():
                    layer_alpha[spec_name] = alpha[:, :, batch_mask, ...]
            else:
                raise ValueError("Unexpected alpha structure.")
        return self

    def mask_batch_dim(self, batch_mask) -> "AlphaValueData":
        """Return a new AlphaValueData with the batch dimension masked."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha[:, :, batch_mask, ...]
                    for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict

    def to(self, *, device=None, dtype=None, non_blocking=False) -> "AlphaValueData":
        """Return a new AlphaValueData with tensors moved to the specified device and/or dtype."""
        new_dict = AlphaValueData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {
                    spec_name: alpha.to(
                        device=device, dtype=dtype, non_blocking=non_blocking
                    )
                    for spec_name, alpha in layer_alpha.items()
                }
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict


@dataclass
class AlphaFullInfoData(DictLikeMixIn):
    """
    A batch of Alpha **values and related auxilaries**, used during .build() stage.

    For each layer, store its full information. Used during initialization stage.
    (e.g. layer.alpha/layer.alpha_lookup_idx/layer.alpha_indices/layer.init)

    Format:
    {activation_layer -> {
                final_layer -> {alpha-> tensor(2,1,batch,fixed_number), alpha_lookup_idx -> ..., ... -> ...}
                ... -> ...
            }
        ...
    }
    """

    _data: dict

    @staticmethod
    def from_net_part(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network with part scope (alpha_start_nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        net.alpha_drop_unused()
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        # Phase probing (see LiRPANet.alpha_drop_unused): while extra
        # intermediate-start-node alphas are retained on the net for the
        # prober, alpha_drop_unused() above no longer removes them, so the
        # extraction must filter down to the normal part scope
        # (alpha_start_nodes) explicitly to keep the extracted data
        # identical to the non-probing case. No-op otherwise.
        keep = (set(net.alpha_start_nodes)
                if getattr(net, 'phase_probing_keep_alpha_nodes', None)
                else None)

        for m in optimizable_activations:
            if not move:
                data = m.dump_alpha()
            else:
                data = m.pop_alpha()
            if keep is not None:
                for field in ('alpha', 'alpha_lookup_idx'):
                    if isinstance(data.get(field), dict):
                        data[field] = type(data[field])(
                            (k, v) for k, v in data[field].items()
                            if k in keep)
            raw_alpha_by_layer[m.name] = data

        return AlphaFullInfoData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net_full(
        net: "LiRPANet",
        *,
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network with full scope (all nodes).
        Args:
            net: The network to extract from.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        raw_alpha_by_layer = {}
        optimizable_activations = net.net.get_enabled_opt_act()

        for m in optimizable_activations:
            if not move:
                raw_alpha_by_layer[m.name] = m.dump_alpha()
            else:
                raw_alpha_by_layer[m.name] = m.pop_alpha()

        return AlphaFullInfoData(_data=raw_alpha_by_layer)

    @staticmethod
    def from_net(
        net: "LiRPANet",
        *,
        starting_node_scope: Literal["part", "all"],
        move: bool = False,
    ) -> "AlphaFullInfoData":
        """
        Extract full alpha info from the network, including alpha tensor and other related info.
        Args:
            net: The network to extract from.
            starting_node_scope: The scope of starting nodes to extract. Additionally, if "part", call net.alpha_drop_unused() first.
            move: If True, remove the alpha values from the network after extraction.
        Returns:
            An AlphaFullInfoData instance containing the extracted alpha values
        """
        if starting_node_scope == "part":
            return AlphaFullInfoData.from_net_part(net, move=move)
        elif starting_node_scope == "all":
            return AlphaFullInfoData.from_net_full(net, move=move)
        else:
            raise ValueError(f"Unknown starting_node_scope: {starting_node_scope}")

    def to(self, *, device=None, dtype=None, non_blocking=False) -> "AlphaFullInfoData":
        """Return a new AlphaFullInfoData with tensors moved to the specified device and/or dtype."""
        new_dict = AlphaFullInfoData(_data={})
        for layer_name, layer_alpha in self.items():
            if isinstance(layer_alpha, dict):
                new_dict[layer_name] = {}
                for spec_name, spec_content in layer_alpha.items():
                    if isinstance(spec_content, torch.Tensor):
                        new_dict[layer_name][spec_name] = spec_content.to(
                            device=device, dtype=dtype, non_blocking=non_blocking
                        )
                    else:
                        new_dict[layer_name][spec_name] = spec_content
            else:
                raise ValueError("Unexpected alpha structure.")
        return new_dict
