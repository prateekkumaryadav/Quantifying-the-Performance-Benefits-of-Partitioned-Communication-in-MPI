import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
exp1 = df[df["exp"] == "exp1"]
exp2 = df[df["exp"] == "exp2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Replicating Figure 5/7 (Small Message Overhead)
ax1.plot(exp1["msg_size"], exp1["single"], "o-", label="Pt2Pt single (Bulk Sync)", color="tab:orange", linewidth=2)
ax1.plot(exp1["msg_size"], exp1["many"], "s-", label="Pt2Pt many (Pipelined)", color="tab:green", linewidth=2)
ax1.plot(exp1["msg_size"], exp1["part"], "^--", label="Pt2Pt part (Open MPI 5.0)", color="tab:blue", linewidth=2)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Message Size [Bytes]")
ax1.set_ylabel("Communication Time [us] (Lower is Better)")
ax1.set_title("Exp 1: Small Message Contention (32 Partitions, gamma=0)")
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# Plot 2: Replicating Figure 8 (Early-Bird Bandwidth Crossover)
ax2.plot(exp2["msg_size"], exp2["single"], "o-", label="Pt2Pt single (Bulk Sync)", color="tab:orange", linewidth=2)
ax2.plot(exp2["msg_size"], exp2["many"], "s-", label="Pt2Pt many (Early-Bird Pipelined)", color="tab:green", linewidth=2)
ax2.plot(exp2["msg_size"], exp2["part"], "^--", label="Pt2Pt part (Open MPI 5.0)", color="tab:blue", linewidth=2)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Message Size [Bytes]")
ax2.set_ylabel("Perceived Bandwidth [GB/s] (Higher is Better)")
ax2.set_title("Exp 2: Large Message Early-Bird Gain (4 Threads, gamma=30 us/MB)")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("m4_replication_plots.png", dpi=300)
print("Saved m4_replication_plots.png successfully!")