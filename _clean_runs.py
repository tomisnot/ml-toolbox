import os, shutil, sys
R = r'D:\code-nosync\Mechine learning\runs'
KEEP = {'custom_network-4272143-43',      # 扇区分类器 0.9684（auto_v8j 直接按路径装载）
        'custom_network-5840349-2026',    # V8win_ktilde_12ep（fid 0.9217，V8 基线）
        'custom_network-7450343-2026',    # V8J_softsec_n12_15ep（当前联合模型 0.9316）
        'custom_network-7504857-2026',    # V8e_opt_30ep（V8 全量基线 0.9232）
        'custom_network-8506161-31',      # V8e_n16only_v2（n16 专项）
        'custom_network-7430342-2026'}    # V8e_dedup_7940
apply = '--apply' in sys.argv
freed = 0.0; ndel = 0; kept = 0
for d in sorted(os.listdir(R)):
    p = os.path.join(R, d)
    if not os.path.isdir(p):
        continue
    if d in KEEP:
        kept += 1; continue
    for sub in ('checkpoint', 'tb'):
        q = os.path.join(p, sub)
        if os.path.isdir(q):
            freed += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(q) for f in fs)
            if apply: shutil.rmtree(q, ignore_errors=True)
    q = os.path.join(p, 'artifacts.npz')
    if os.path.isfile(q):
        freed += os.path.getsize(q)
        if apply: os.remove(q)
    ndel += 1
print('%s：清理 %d 个 run（保留 %d 个）｜释放 %.1f MB'
      % ('已执行' if apply else '预演', ndel, kept, freed / 1e6))
