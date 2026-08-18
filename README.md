# mUTR-Construct
A Conditional Structure-Aware Generative Transformer for Multi-Objective Design of m1Ψ-Modified RNA 5’ UTRs


File generate_compare_control_real_struct_pareto_smart5utr_v3.py:

Generate 500 sequences per scenario with a trained hybrid RL+GC+MFE+structure generator,
run a Smart5UTR-only baseline for the same scenarios, and aggregate comparison statistics.

This script is intentionally mode-based. 

Typical usage:

1) Show all scenarios:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py --mode list_scenarios --scenario_set full

2) Run one scenario by SLURM_ARRAY_TASK_ID:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py \
     --mode run_scenario \
     --scenario_set full \
     --generator_model hybrid_struct_besthp_training/hybrid_struct_besthp_model.pt \
     --train_csv benchmark_sequences.csv \
     --smart5utr_model ../models/Smart5UTR/Smart5UTR_egfp_m1pseudo2_Model.h5 \
     --smart5utr_scaler ../models/egfp_m1pseudo2.scaler \
     --modification_json rna_mod_n1methylpseudouridine_parameters.json \
     --n_results 500 \
     --out_dir scenario_outputs_struct_besthp

3) Aggregate after all array tasks finish:
   python generate_compare_control_real_hybrid_struct_pareto_smart5utr.py \
     --mode aggregate \
     --scenario_set full \
     --out_dir scenario_outputs_struct_besthp
