import torch
from typing import Optional
import os
from torch.utils.cpp_extension import load
CHUNK_LEN = int(os.environ["CHUNK_LEN"])
HEAD_SIZE = int(os.environ["RWKV_REWARD_MODEL_HEAD_SIZE_A"])
full_parent_dir = os.path.dirname(os.path.abspath(__file__))
flags = [
    "-res-usage",
    f"-D_C_={HEAD_SIZE}",
    f"-D_CHUNK_LEN_={CHUNK_LEN}",
    "--use_fast_math",
    "-O3",
    "-Xptxas -O3",
    "--extra-device-vectorization",
]
load(
    name="wind_backstepping",
    sources=[f"{full_parent_dir}/cuda/wkv7_cuda.cu", f"{full_parent_dir}/cuda/wkv7_op.cpp"],
    is_python_module=False,
    verbose=True,
    extra_cuda_cflags=flags,
)


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(
            B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
        )
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype == torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(
            w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
        )
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)
