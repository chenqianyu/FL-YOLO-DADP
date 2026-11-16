#!/usr/bin/env bash
# Re-run the experiments behind Table 1 and Fig. 3 of the paper, in order.
# Expects the data layout described in README.md ("Data"). Already finished runs
# are skipped, so the script can be re-started after an interruption.
#
#   bash scripts/reproduce_paper.sh
#
# Generative baselines (notebooks) and Fig. 2 (Eigen-CAM notebook) are manual
# steps; see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=20
GPU_FRACTION=${GPU_FRACTION:-0.3}   # GPU share per FL client (0.3 => 3 clients in parallel)

train() {  # train <name> <data.yaml> <epochs> [extra args...]
    local name=$1 data=$2 epochs=$3; shift 3
    if [ -f "runs/baselines/$name/weights/best.pt" ]; then echo "skip $name"; return; fi
    python baselines/train_yolo.py --data "$data" --epochs "$epochs" --name "$name" "$@"
}

fl() {  # fl <run_name> [run_fl.py args...]
    local name=$1; shift
    if [ -f "runs/fl/$name/round$ROUNDS/FL_aggregated_round_$ROUNDS.pt" ]; then echo "skip $name"; return; fi
    python fl/run_fl.py --run-name "$name" --num-rounds "$ROUNDS" --num-gpus "$GPU_FRACTION" --overwrite "$@"
}

# 1. Centralised benchmark (Table 1 "YOLOv8n") and the FL round-1 seed
train benchmark_yolov8n data/pooled/data.yaml 20
train seed_agent1_epoch1 data/agents/agent_1/data.yaml 1 --export weights/seed_yolov8n_agent1_epoch1.pt
[ -f weights/seed_yolov8n_agent1_epoch1.pt ] || cp runs/baselines/seed_agent1_epoch1/weights/best.pt weights/seed_yolov8n_agent1_epoch1.pt

# 2. Local-only models, one per agent (Fig. 3 "Benchmark (Avg 3 agents)"; agent 2 also used in Fig. 2)
for k in 0 1 2; do train "local_agent${k}_epoch20" "data/agents/agent_$k/data.yaml" 20; done

# 3. Federated baselines without DP (Table 1)
fl FedAvg_3agents_round20_epoch1    --strategy FedAvg
fl FedMedian_3agents_round20_epoch1 --strategy FedMedian

# 4. LDP / DADP grid (Fig. 3). FedMedian + DADP, sigma_0=1e-4, C=1 is the Table 1 model.
for nm in 0.0001 0.001; do
  for cn in 1 10; do
    for strategy in FedAvg FedMedian; do
      for dp in ldp dadp; do
        DP=$(echo "$dp" | tr a-z A-Z)
        fl "${strategy}_${DP}_round${ROUNDS}_epoch1_nm${nm}_cn${cn}" \
           --strategy "$strategy" --dp "$dp" --noise-multiplier "$nm" --clipping-norm "$cn"
      done
    done
  done
done

# 5. DADP run used for the Eigen-CAM visualisation (Fig. 2)
fl FedMedian_DADP_round20_epoch1_nm0_cn10 --strategy FedMedian --dp dadp --noise-multiplier 0 --clipping-norm 10

# 6. Table 1 (add the synthetic_* models once the generative notebooks have been run)
MODELS=(
  --model "YOLOv8n=runs/baselines/benchmark_yolov8n/weights/best.pt"
  --model "FedAvg-YOLO=runs/fl/FedAvg_3agents_round20_epoch1/round20/FL_aggregated_round_20.pt"
  --model "FedMedian-YOLO=runs/fl/FedMedian_3agents_round20_epoch1/round20/FL_aggregated_round_20.pt"
  --model "FL-YOLO-DADP=runs/fl/FedMedian_DADP_round20_epoch1_nm0.0001_cn1/round20/FL_aggregated_round_20.pt"
)
for m in encoder-decoder GANs DDPM controlnet_img2img; do
  w="runs/baselines/synthetic_$m/weights/best.pt"
  [ -f "$w" ] && MODELS+=(--model "$m=$w")
done
python evaluation/evaluate.py --data data/agents/agent_0/data.yaml --split test --out results/table1.csv "${MODELS[@]}"

# 7. Fig. 3
python evaluation/plot_fl_curves.py --runs-root runs/fl --out results/fig3.png \
  --benchmark runs/baselines/local_agent{0,1,2}_epoch20/results.csv
