# ~/vf-cot/scripts/visualize_alpha.py
# 실행: cd ~/vf-cot/scripts && python visualize_alpha.py

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── 데이터 로드 ───────────────────────────────────────────
with open("results/main_exp/vf_score_alpha_search_v2.json") as f:
    results = json.load(f)

alphas      = [r["alpha"]      for r in results]
spearman_rs = [r["spearman_r"] for r in results]
pearson_rs  = [r["pearson_r"]  for r in results]
p_values    = [r["spearman_p"] for r in results]

best_alpha = max(results, key=lambda x: x["spearman_r"])["alpha"]
best_rho   = max(results, key=lambda x: x["spearman_r"])["spearman_r"]

# ── 그래프 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle("VF-Score Alpha Search: Correlation with Human Judgments",
             fontsize=14, fontweight="bold", y=0.98)

# ── 그래프 1: Spearman & Pearson 상관관계 ─────────────────
ax1 = axes[0]
ax1.plot(alphas, spearman_rs, "o-", color="#4A90E2", linewidth=2,
         markersize=5, label="Spearman ρ")
ax1.plot(alphas, pearson_rs,  "s--", color="#E67E22", linewidth=2,
         markersize=5, label="Pearson r")

# 최적 alpha 표시
ax1.axvline(x=best_alpha, color="#27AE60", linestyle="--",
            linewidth=1.5, alpha=0.8)
ax1.scatter([best_alpha], [best_rho], color="#27AE60",
            s=120, zorder=5)
ax1.annotate(f"Best α={best_alpha}\nρ={best_rho:.4f}",
             xy=(best_alpha, best_rho),
             xytext=(best_alpha + 0.08, best_rho - 0.04),
             fontsize=10, color="#27AE60",
             arrowprops=dict(arrowstyle="->", color="#27AE60"))

# 0 기준선
ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

# 영역 색칠
ax1.axvspan(0.0, 0.25, alpha=0.08, color="#27AE60",
            label="Faithfulness-dominant zone")
ax1.axvspan(0.75, 1.0, alpha=0.08, color="#E74C3C",
            label="Similarity-dominant zone")

ax1.set_xlabel("Alpha (α)", fontsize=11)
ax1.set_ylabel("Correlation Coefficient", fontsize=11)
ax1.set_title("Spearman ρ & Pearson r vs Alpha", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.45, 0.45)

# 해석 텍스트
ax1.text(0.12, 0.36, "Faithfulness\ndominant\n(α → 0)",
         fontsize=9, color="#27AE60", ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#27AE60", alpha=0.8))
ax1.text(0.88, 0.36, "Similarity\ndominant\n(α → 1)",
         fontsize=9, color="#E74C3C", ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#E74C3C", alpha=0.8))

# ── 그래프 2: p-value 분포 ────────────────────────────────
ax2 = axes[1]
colors = ["#27AE60" if p < 0.05 else "#E74C3C" for p in p_values]
bars = ax2.bar(alphas, p_values, width=0.04, color=colors, alpha=0.7)

# p=0.05 기준선
ax2.axhline(y=0.05, color="#E74C3C", linestyle="--",
            linewidth=1.5, label="p = 0.05 (significance threshold)")

# 최적 alpha 표시
ax2.axvline(x=best_alpha, color="#27AE60", linestyle="--",
            linewidth=1.5, alpha=0.8, label=f"Best α = {best_alpha}")

ax2.set_xlabel("Alpha (α)", fontsize=11)
ax2.set_ylabel("p-value", fontsize=11)
ax2.set_title("Statistical Significance (p-value) vs Alpha", fontsize=12)

sig_patch   = mpatches.Patch(color="#27AE60", alpha=0.7,
                              label="p < 0.05 (significant)")
insig_patch = mpatches.Patch(color="#E74C3C", alpha=0.7,
                              label="p ≥ 0.05 (not significant)")
ax2.legend(handles=[sig_patch, insig_patch] +
           [plt.Line2D([0], [0], color="#E74C3C", linestyle="--",
                       label="p = 0.05 threshold"),
            plt.Line2D([0], [0], color="#27AE60", linestyle="--",
                       label=f"Best α = {best_alpha}")],
           fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_xlim(-0.02, 1.02)

plt.tight_layout()
Path("results/main_exp").mkdir(parents=True, exist_ok=True)
out_path = "results/main_exp/alpha_search_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장 완료 → {out_path}")
plt.close()

# ── 텍스트 요약 ───────────────────────────────────────────
print("\n=== 분석 요약 ===")
print(f"최적 alpha:       {best_alpha}")
print(f"최고 Spearman ρ:  {best_rho:.4f}")
print(f"해석: Faithfulness {round((1-best_alpha)*100)}% + Similarity {round(best_alpha*100)}%")

sig_alphas = [r["alpha"] for r in results if r["spearman_p"] < 0.05]
print(f"\n통계적으로 유의미한 alpha 범위: {min(sig_alphas):.2f} ~ {max(sig_alphas):.2f}")

pos_alphas = [r["alpha"] for r in results if r["spearman_r"] > 0]
print(f"양의 상관관계 alpha 범위:       {min(pos_alphas):.2f} ~ {max(pos_alphas):.2f}")