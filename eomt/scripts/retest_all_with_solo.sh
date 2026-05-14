#!/usr/bin/env bash
# Re-test every run in eomt/runs with the new solo_map metric.
# Writes test_results_seq.txt and test_results_single.txt into each run dir.
set -e
cd /home/dsa/new_seg_final/eomt
PY=/home/dsa/miniconda3/envs/new_seg_final/bin/python

run_test () {
  local name="$1" version="$2" ckpt="$3" cfg="$4" filename="$5"
  echo "=========================================================="
  echo ">>> ${name}/${version}  cfg=${cfg}  out=${filename}"
  echo "=========================================================="
  $PY main.py test --config "${cfg}" --ckpt_path "${ckpt}" \
    --trainer.logger.init_args.name "${name}" \
    --trainer.logger.init_args.version "${version}" \
    --model.init_args.test_results_filename "${filename}" \
    2>&1 | tail -5
}

# ---------------- 2D vanilla ----------------
RN=coronary_instance_eomt_small_512_dinov2
CK=runs/${RN}/version_0/checkpoints/best.ckpt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/instance/test_2d_model_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/instance/test_2d_model_on_single_dataset.yaml test_results_single.txt

# ---------------- 2D new_augs ----------------
RN=coronary_instance_eomt_small_512_dinov2_new_augs
CK=runs/${RN}/version_1/checkpoints/best.ckpt
run_test "${RN}" version_1 "${CK}" configs/dinov2/coronary/instance/test_2d_model_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" version_1 "${CK}" configs/dinov2/coronary/instance/test_2d_model_on_single_dataset.yaml test_results_single.txt

# ---------------- 2D skelrecall ----------------
RN=coronary_instance_eomt_small_512_dinov2_skelrecall
CK=runs/${RN}/checkpoints/best.ckpt
run_test "${RN}" test_seq    "${CK}" configs/dinov2/coronary/instance/test_skelrecall_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" test_single "${CK}" configs/dinov2/coronary/instance/test_skelrecall_on_single_dataset.yaml test_results_single.txt

# ---------------- 2D skelrecall_connect ----------------
RN=coronary_instance_eomt_small_512_dinov2_skelrecall_connect
CK=runs/${RN}/checkpoints/best.ckpt
run_test "${RN}" test_seq    "${CK}" configs/dinov2/coronary/instance/test_skelrecall_connect_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" test_single "${CK}" configs/dinov2/coronary/instance/test_skelrecall_connect_on_single_dataset.yaml test_results_single.txt

# ---------------- Video vanilla ----------------
RN=coronary_video_eomt_small_512_dinov2
CK=runs/${RN}/version_0/checkpoints/best.ckpt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/video_instance/test_video_model_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/video_instance/test_video_model_on_single_dataset.yaml test_results_single.txt

# ---------------- Video skelrecall_connect ----------------
RN=coronary_video_eomt_small_512_dinov2_skelrecall_connect
CK=runs/${RN}/version_0/checkpoints/best.ckpt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/video_instance/test_video_skelrecall_connect_on_seq_dataset.yaml    test_results_seq.txt
run_test "${RN}" version_0 "${CK}" configs/dinov2/coronary/video_instance/test_video_skelrecall_connect_on_single_dataset.yaml test_results_single.txt

echo "ALL DONE"
