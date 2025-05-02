coop_testtime: 包含了 coop 的结果: (82.69, 63.29, 71.70)

VPT: 包含了 vpt-shallow 的结果 (prompt length = 4): (80.17, 70.06, 74.78)
VPT_testtime: 包含了 vpt-mix 的结果 (prompt length = 4): (82.08, 69.10, 75.03)
VPT_deep: 包含了 vpt-deep 的结果 (prompt length = 4): (83.64, 67.31, 74.59)

Unified: 包含了 vpt-deep + coop, seperate 的结果
Unified_v2: 包含了 vpt-mix + coop, seperate 的结果
Unified_v3: 包含了 vpt-deep + coop, shared 的结果 (10e: 80.40, 74.24, 77.20; 200e: 84.35, 64.86, 73.33)
Unified_v4: 包含了 vpt-deep + coop, DETR encoder 的结果 (200e: 74.56)
Unified_v5: 包含了 vpt-deep + coop self-attn 的结果 (10e: 76.99; 200e: 74.20)