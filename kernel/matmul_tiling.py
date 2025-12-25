import triton 
import triton.language as tl
import torch
import math

from jaxtyping import Float
from torch import Tensor
from einops import rearrange

BLOCK_SZIE = 1024

TILE_SIZE_M  = math.isqrt(BLOCK_SZIE)
TILE_SIZE_N  = math.isqrt(BLOCK_SZIE)

# 计算 offset 的函数: 
# 这个函数是有语义的:  ld_row, ld_col 就代表了 tile 的 shape
# OFFSET_BLOCK = lambda row,col,shape,ld: (row) * (ld * shape[1]) + col * shape[0]

@triton.jit
def OFFSET_BLOCK(row, col, shape0, shape1, ld):
    return row * (ld * shape0) + col * shape1


@triton.jit
def matmul_tiled(
    a,
    b,
    c,
    stride_ab: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_cb: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr
):
    pid_batch = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    batch_offset_a = a + OFFSET_BLOCK(pid_batch,0, 1,stride_ab, stride_ab)
    batch_offset_b = b + OFFSET_BLOCK(pid_batch,0, 1,stride_bb, stride_bb)
    batch_offset_c = c + OFFSET_BLOCK(pid_batch,0, 1,stride_cb, stride_cb)
    
    a_ptr = batch_offset_a + OFFSET_BLOCK(m,0, TILE_M,K,      K)
    b_ptr = batch_offset_b + OFFSET_BLOCK(0,n, K,TILE_N,      N)
    c_ptr = batch_offset_c + OFFSET_BLOCK(m,n, TILE_M,TILE_N, N)

    
    psum_mat = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    offset_m = tl.arange(0,TILE_M)
    offset_n = tl.arange(0,TILE_N)
    

    value_a_mask = (m * TILE_M + offset_m[:,None]) < M
    value_b_mask = (n * TILE_N + offset_n[None,:]) < N
    
    for i in range(0,K):
        value_a_ptr  = a_ptr + OFFSET_BLOCK( offset_m[:,None],i, 1,1, K)
        value_a      = tl.load(pointer=value_a_ptr,mask = value_a_mask)     # [TILE_M,1] 
        
        value_b_ptr  = b_ptr + OFFSET_BLOCK(i, offset_n[None,:], 1,1, N)
        value_b      = tl.load(pointer=value_b_ptr,mask = value_b_mask)     # [1,TILE_N]
        
        psum_mat    += value_a * value_b                                    # [TILE_M, TILE_N]
    
    value_c_ptr  = c_ptr + OFFSET_BLOCK(offset_m[:,None], offset_n[None,:], 1,1,  N)
    value_c_mask_m = (m * TILE_M + offset_m[:,None]) < M
    value_c_mask_n = (n * TILE_N + offset_n[None,:]) < N
    value_c_mask   = value_c_mask_m & value_c_mask_n
    
    tl.store(pointer=value_c_ptr, mask = value_c_mask, value=psum_mat)
    

# One block computing the block-size C
def matmul(
    a: Float[Tensor, "... i j"],
    b: Float[Tensor, "... j k"],
):
    assert a.is_cuda and b.is_cuda
    
    batch_shape = a.shape[:-2]
    a_3d = rearrange(a, "... i k -> (...) i k")
    b_3d = rearrange(b, "... k j -> (...) k j")

    batch_size, M, K = a_3d.shape
    batch_size, K, N = b_3d.shape

    c_3d = torch.zeros(batch_size, M, N, device=a.device, dtype=a.dtype)

    grid = (
        batch_size,
        triton.cdiv(M, TILE_SIZE_M),
        triton.cdiv(N, TILE_SIZE_N)
    )

    matmul_tiled[grid](
        a = a_3d, b = b_3d, c = c_3d,
        stride_ab = a_3d.stride(0),
        stride_bb = b_3d.stride(0),
        stride_cb = c_3d.stride(0),
        M = M, K = K, N = N,
        TILE_M = TILE_SIZE_M,
        TILE_N = TILE_SIZE_N
    )

    return rearrange(c_3d, "(batch) i j -> batch i j", batch = math.prod(batch_shape))


# if __name__ == "__main__":
#     A = torch.rand(128,512,2048).cuda()
#     B = torch.rand(128,2048,1024).cuda()

#     Y          = matmul(A,B)
#     OFFICIAL_Y = torch.matmul(A,B)
#     error = (OFFICIAL_Y - Y).mean()
#     print(f"Mean absolute error: {error:.10f}")


def stress_test():
    """压力测试：更多边界情况"""
    print("\n" + "="*70)
    print("STRESS TESTS")
    print("="*70)
    
    test_cases = [
        # (batch, M, K, N, description)
        (1, 1, 1, 1, "Minimal size"),
        (1, 32, 32, 32, "Small square"),
        (5, 100, 200, 150, "Irregular sizes"),
        (10, 999, 1001, 997, "Prime-like sizes"),
        (2, 1024, 2048, 512, "Large K"),
        (128, 7, 13, 11, "Small with large batch"),
    ]
    
    all_perfect = True
    
    for batch, M, K, N, desc in test_cases:
        A = torch.rand(batch, M, K).cuda()
        B = torch.rand(batch, K, N).cuda()
        
        Y = matmul(A, B)
        REF = torch.matmul(A, B)
        
        max_err = (Y - REF).abs().max().item()
        exact_match = (Y == REF).all().item()
        
        status = "✓ PERFECT" if exact_match else f"✗ Error: {max_err:.2e}"
        print(f"   {desc:25s} [{batch:3d}, {M:4d}, {K:4d}, {N:4d}]: {status}")
        
        if not exact_match:
            all_perfect = False
    
    print(f"\n   {'='*66}")
    if all_perfect:
        print(f"   🎉 ALL TESTS PERFECT! Your implementation is FLAWLESS! 🎉")
    else:
        print(f"   ⚠️  Some tests showed differences (still likely correct)")
    print(f"   {'='*66}")



if __name__ == "__main__":  
    print("="*70)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("="*70)
    
    A = torch.rand(128, 512, 2048).cuda()  
    B = torch.rand(128, 2048, 1024).cuda()  
  
    Y = matmul(A, B)  
    OFFICIAL_Y = torch.matmul(A, B)  
    
    # 1. 基本误差统计
    print("\n1. BASIC ERROR STATISTICS:")
    abs_diff = (OFFICIAL_Y - Y).abs()
    print(f"   Mean absolute error: {abs_diff.mean():.15e}")
    print(f"   Max absolute error:  {abs_diff.max():.15e}")
    print(f"   Min absolute error:  {abs_diff.min():.15e}")
    print(f"   Std absolute error:  {abs_diff.std():.15e}")
    
    # 2. 相对误差
    print("\n2. RELATIVE ERROR:")
    rel_diff = abs_diff / (OFFICIAL_Y.abs() + 1e-10)
    print(f"   Mean relative error: {rel_diff.mean():.15e}")
    print(f"   Max relative error:  {rel_diff.max():.15e}")
    
    # 3. 完全相等的元素
    print("\n3. EXACT EQUALITY:")
    exact_match = (Y == OFFICIAL_Y).sum().item()
    total = Y.numel()
    print(f"   Exactly equal elements: {exact_match:,} / {total:,}")
    print(f"   Percentage: {100.0 * exact_match / total:.6f}%")
    
    # 4. 误差分布
    print("\n4. ERROR DISTRIBUTION:")
    thresholds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for thresh in thresholds:
        count = (abs_diff < thresh).sum().item()
        pct = 100.0 * count / total
        print(f"   < {thresh:.0e}: {count:,} elements ({pct:.2f}%)")
    
    # 5. torch.allclose 测试
    print("\n5. TORCH.ALLCLOSE TESTS:")
    test_params = [
        (1e-8, 1e-8),
        (1e-7, 1e-7),
        (1e-6, 1e-6),
        (1e-5, 1e-5),
        (1e-4, 1e-4),
    ]
    for rtol, atol in test_params:
        result = torch.allclose(Y, OFFICIAL_Y, rtol=rtol, atol=atol)
        print(f"   rtol={rtol:.0e}, atol={atol:.0e}: {result}")
    
    # 6. 检查数据类型和设备
    print("\n6. DATA INFO:")
    print(f"   Y dtype: {Y.dtype}, device: {Y.device}")
    print(f"   OFFICIAL_Y dtype: {OFFICIAL_Y.dtype}, device: {OFFICIAL_Y.device}")
    print(f"   Y shape: {Y.shape}")
    print(f"   OFFICIAL_Y shape: {OFFICIAL_Y.shape}")
    
    # 7. 采样检查几个具体值
    print("\n7. SAMPLE VALUES (first 5 elements of first batch):")
    print(f"   Y:          {Y[0, 0, :5]}")
    print(f"   OFFICIAL_Y: {OFFICIAL_Y[0, 0, :5]}")
    print(f"   Difference: {(Y - OFFICIAL_Y)[0, 0, :5]}")
    
    # 8. 检查是否有 NaN 或 Inf
    print("\n8. VALIDITY CHECK:")
    print(f"   Y has NaN: {torch.isnan(Y).any().item()}")
    print(f"   Y has Inf: {torch.isinf(Y).any().item()}")
    print(f"   OFFICIAL_Y has NaN: {torch.isnan(OFFICIAL_Y).any().item()}")
    print(f"   OFFICIAL_Y has Inf: {torch.isinf(OFFICIAL_Y).any().item()}")
    
    # 9. 多次运行测试（确保不是偶然）
    print("\n9. MULTIPLE RUNS TEST:")
    for i in range(3):
        A_test = torch.rand(4, 64, 128).cuda()
        B_test = torch.rand(4, 128, 64).cuda()
        Y_test = matmul(A_test, B_test)
        OFFICIAL_Y_test = torch.matmul(A_test, B_test)
        error = (Y_test - OFFICIAL_Y_test).abs().max()
        print(f"   Run {i+1}: max error = {error:.15e}")
    
    print("\n" + "="*70)    

    stress_test()