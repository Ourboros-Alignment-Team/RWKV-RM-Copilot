import torch
import copy


class BlockState:
    def __init__(
        self,
        time_mix_state: tuple[torch.Tensor, torch.Tensor],
        channel_mix_state: torch.Tensor,
    ):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:
    def __init__(self, shift_states, wkv_states):
        self.shift_states = shift_states
        self.wkv_states = wkv_states

    def clone(self):
        return BlockStateList(
            self.shift_states.detach().clone(), self.wkv_states.detach().clone()
        )

    @torch.no_grad()
    def duplicate(self, times):
        wkv_states_list = []
        shift_states_list = []
        for _ in range(times):
            wkv_states_list.append(copy.deepcopy(self.wkv_states))
            shift_states_list.append(copy.deepcopy(self.shift_states))
        return BlockStateList(
            shift_states=torch.cat(shift_states_list, dim=2),
            wkv_states=torch.cat(wkv_states_list, dim=1),
        )

    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result
    
    @staticmethod
    def create_like(other, batch_size=None):
        N, B, n_head, head_size, head_size = other.wkv_states.size()
        _, _, _, C = other.shift_states.size()
        if batch_size is not None:
            B = batch_size
        return BlockStateList.create(
            N,
            B,
            C,
            n_head,
            head_size,
            other.wkv_states.device,
            other.wkv_states.dtype,
        )

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        wkv_states = torch.empty(
            (N, B, n_head, head_size, head_size), device=device, dtype=torch.float
        )
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    @staticmethod
    def merge(s1, s2, s1_part: float = 0.5, s2_part: float = 0.5):
        wkv_states = s1.wkv_states * s1_part + s2.wkv_states * s2_part
        shift_states = s1.shift_states * s1.s1_part + s2.shift_states * s2_part
        return BlockStateList(shift_states, wkv_states)

    def decay(self, ratio: float = 0.95):
        if ratio == 0:
            return self
        self.wkv_states = self.wkv_states * ratio
        self.shift_states = self.shift_states
        return self

    def __getitem__(self, layer: int):
        if isinstance(layer, int): 
            return BlockState(
                (self.shift_states[layer, 0], self.wkv_states[layer]),
                (self.shift_states[layer, 1]),
            )
        elif isinstance(layer, slice):  # 切片功能
            new_wkv_states = self.wkv_states[layer]
            new_shift_states = self.shift_states[layer]
            return BlockStateList(new_shift_states, new_wkv_states)
        else:
            raise TypeError(f"Invalid index type: {type(layer)}")

    def batchof(self, batch_idx):
        if not (0 <= batch_idx < self.wkv_states.size(1)):
            raise ValueError(
                f"dim_index 超出范围，应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
            )
        return BlockStateList(
            self.shift_states[:, :, batch_idx : batch_idx + 1, :],
            self.wkv_states[:, batch_idx : batch_idx + 1, :, :, :],
        )

    def __add__(self, other):
        if other is None:
            N, B, n_head, head_size, head_size = self.wkv_states.size()
            _, _, _, C = self.shift_states.size()
            return self.__add__(
                BlockStateList.create(
                    N,
                    B,
                    C,
                    n_head,
                    head_size,
                    self.wkv_states.device,
                    self.wkv_states.dtype,
                )
            )
        assert isinstance(other, BlockStateList)
        wkv_states = torch.cat([self.wkv_states, other.wkv_states], dim=1)
        shift_states = torch.cat([self.shift_states, other.shift_states], dim=2)
        return BlockStateList(shift_states, wkv_states)

    def remove_at(self, batch_idx):
        if not (0 <= batch_idx < self.wkv_states.size(1)):
            raise ValueError(
                f"dim_index 超出范围，应在 [0, batch_size: {self.wkv_states.size(1) - 1}] 之间。"
            )
        new_wkv_states = torch.cat(
            (
                self.wkv_states[:, :batch_idx, :, :, :],
                self.wkv_states[:, batch_idx + 1 :, :, :, :],
            ),
            dim=1,
        )
        new_shift_states = torch.cat(
            (
                self.shift_states[:, :, :batch_idx, :],
                self.shift_states[:, :, batch_idx + 1 :, :],
            ),
            dim=2,
        )
        return BlockStateList(new_shift_states, new_wkv_states)

    def unbind(self):
        _wkv = [x.unsqueeze(1) for x in torch.unbind(self.wkv_states, dim=1)]
        _shift = [x.unsqueeze(2) for x in torch.unbind(self.shift_states, dim=2)]
        res = [
            BlockStateList(shift_states, wkv_states)
            for wkv_states, shift_states in zip(_wkv, _shift)
        ]
        return res

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

    def cpu(self):
        wkv_states = self.wkv_states.detach().cpu()
        shift_states = self.shift_states.detach().cpu()
        return BlockStateList(shift_states, wkv_states)

    def cuda(self):
        wkv_states = self.wkv_states.to("cuda")
        shift_states = self.shift_states.to("cuda")
        return BlockStateList(shift_states, wkv_states)

    def to(self, device):
        wkv_states = self.wkv_states.to(device)
        shift_states = self.shift_states.to(device)
        return BlockStateList(shift_states, wkv_states)

    @classmethod
    def save(cls, item, path):
        item = item.cpu()
        data = {"wkv_states": item.wkv_states, "shift_states": item.shift_states}
        torch.save(data, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu")
        wkv_states = data["wkv_states"]
        shift_states = data["shift_states"]
        item = cls(shift_states, wkv_states)
        return item
