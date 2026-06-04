# ~/vf-cot/scripts/analyze_components.py
# 실행: cd ~/vf-cot/scripts && python analyze_components.py

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

# ── 데이터 로드 ───────────────────────────────────────────
with open("results/main_exp/final_vf_scores.json") as f:
    data = json.load(f)

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]
CF_COLORS = {
    "semantic_swap":  "#e74c3c",
    "attribute_flip": "#e67e22",
    "random":         "#8e44ad",
    "masked":         "#2c3e50",
}
CF_LABELS = {
    "semantic_swap":  "Semantic Swap",
    "attribute_flip": "Attribute Flip",
    "random":         "Random",
    "masked":         "Masked",
}

def normalize(score):
    return (score - 1) / (5 - 1)

# ── 컴포넌트 추출 ─────────────────────────────────────────
# (1 - sim_norm): 높을수록 추론이 달라짐 = faithful
# faith_norm:     높을수록 CoT가 말이 됨 = faithful
components = {cf: {"one_minus_sim": [], "faith": [], "vf_score": []}
              for cf in CF_TYPES}

for r in data:
    cf   = r["cf_type"]
    sim_norm   = normalize(r["sim_score"])
    faith_norm = normalize(r["faith_score"])
    one_minus_sim = 1 - sim_norm

    components[cf]["one_minus_sim"].append(one_minus_sim)
    components[cf]["faith"].append(faith_norm)
    components[cf]["vf_score"].append(r["vf_score"])

# ── 그래프 ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("VF-Score Component Analysis:\n(1-Similarity) vs Faithfulness of Modified Image",
             fontsize=14, fontweight="bold")

# ── 그래프 1: CF 유형별 두 컴포넌트 평균 비교 (Bar chart) ──
ax1 = fig.add_subplot(gs[0, :])
x       = np.arange(len(CF_TYPES))
width   = 0.35
means_sim   = [np.mean(components[cf]["one_minus_sim"]) for cf in CF_TYPES]
means_faith = [np.mean(components[cf]["faith"])         for cf in CF_TYPES]
stds_sim    = [np.std(components[cf]["one_minus_sim"])  for cf in CF_TYPES]
stds_faith  = [np.std(components[cf]["faith"])          for cf in CF_TYPES]

bars1 = ax1.bar(x - width/2, means_sim,   width, yerr=stds_sim,
                label="(1 - Similarity)", color="#4A90E2", alpha=0.8,
                capsize=4)
bars2 = ax1.bar(x + width/2, means_faith, width, yerr=stds_faith,
                label="Faithfulness of Modified Image", color="#E67E22",
                alpha=0.8, capsize=4)

ax1.set_xticks(x)
ax1.set_xticklabels([CF_LABELS[cf] for cf in CF_TYPES], fontsize=11)
ax1.set_ylabel("Normalized Score (0~1)", fontsize=11)
ax1.set_title("Mean Component Scores by CF Type", fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3, axis="y")

# 값 레이블
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# ── 그래프 2~5: CF 유형별 scatter plot ────────────────────
for idx, cf_type in enumerate(CF_TYPES):
    row = (idx // 2) + 1
    col = idx % 2
    ax  = fig.add_subplot(gs[row, col])

    x_vals = components[cf_type]["one_minus_sim"]
    y_vals = components[cf_type]["faith"]
    color  = CF_COLORS[cf_type]

    ax.scatter(x_vals, y_vals, alpha=0.5, color=color, s=30)

    # 상관관계
    r_val, p_val = stats.pearsonr(x_vals, y_vals)

    # 추세선
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), "--", color=color, alpha=0.8, linewidth=1.5)

    # 평균점
    ax.scatter([np.mean(x_vals)], [np.mean(y_vals)],
               color=color, s=120, marker="*", zorder=5,
               label=f"Mean ({np.mean(x_vals):.2f}, {np.mean(y_vals):.2f})")

    ax.set_xlabel("(1 - Similarity)", fontsize=10)
    ax.set_ylabel("Faithfulness", fontsize=10)
    ax.set_title(f"{CF_LABELS[cf_type]}\nr={r_val:.3f}, p={p_val:.4f}",
                 fontsize=11, color=color)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 사분면 구분선
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

Path("results/main_exp").mkdir(parents=True, exist_ok=True)
out_path = "results/main_exp/component_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장 완료 → {out_path}")
plt.close()

# ── 텍스트 요약 ───────────────────────────────────────────
print("\n=== 컴포넌트 분석 요약 ===")
print(f"\n{'CF Type':<16} {'(1-Sim) Mean':>13} {'Faith Mean':>11} {'차이':>8} {'더 중요한 것':>14}")
print("-" * 70)
for cf in CF_TYPES:
    sim_mean   = np.mean(components[cf]["one_minus_sim"])
    faith_mean = np.mean(components[cf]["faith"])
    diff       = abs(sim_mean - faith_mean)
    dominant   = "(1-Sim)" if sim_mean > faith_mean else "Faithfulness"
    print(f"{CF_LABELS[cf]:<16} {sim_mean:>13.4f} {faith_mean:>11.4f} "
          f"{diff:>8.4f} {dominant:>14}")

print("\n=== 전체 평균 ===")
all_sim   = [v for cf in CF_TYPES for v in components[cf]["one_minus_sim"]]
all_faith = [v for cf in CF_TYPES for v in components[cf]["faith"]]
print(f"(1-Similarity) 전체 평균: {np.mean(all_sim):.4f} ± {np.std(all_sim):.4f}")
print(f"Faithfulness   전체 평균: {np.mean(all_faith):.4f} ± {np.std(all_faith):.4f}")

r_val, p_val = stats.pearsonr(all_sim, all_faith)
print(f"\n두 컴포넌트 간 상관관계: r={r_val:.4f}, p={p_val:.4f}")
if abs(r_val) < 0.3:
    print("→ 두 컴포넌트는 독립적으로 다른 정보를 측정함 (좋은 신호)")
else:
    print("→ 두 컴포넌트 간 상관관계 존재 (중복 정보 가능성)")