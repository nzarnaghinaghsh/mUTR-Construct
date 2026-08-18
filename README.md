# mUTR-Construct
A Conditional Structure-Aware Generative Transformer for Multi-Objective Design of m1Ψ-Modified RNA 5’ UTRs

File train_hybrid_struct_generator_besthp.py:

Training-only script for the hybrid conditional generator with four conditioning
features:

    RL + GC + MFE + dot-bracket secondary structure

This is the mode-free/comment-free replacement for the old "comment/uncomment
main section" workflow. It retrains one final model using the best hyper-parameters:

    batch_size = 32
    dropout    = 0.15
    ff_dim     = 256
    d_model    = 64
    epochs     = 8
    layers     = 3
    lr         = 1e-4

It also supports optional train/validation split so you can confirm the loss,
but it always saves a final full-data model by default.

Expected CSV columns:
    Required:
        sequence  OR  utr

    Optional:
        rl / RL / mrl / MRL / target_rl
        gc / GC / target_gc
        mfe / MFE / target_mfe
        structure / target_structure

If MFE or structure are missing, they are computed using ViennaRNA. If you pass
--modification_json, the script loads that JSON before folding.

Example:
    python train_hybrid_struct_generator_besthp.py \
      --train_csv benchmark_sequences.csv \
      --output_dir hybrid_struct_besthp_training \
      --out_model hybrid_struct_besthp_final.pt \
      --modification_json rna_mod_n1methylpseudouridine_parameters.json

Optional validation split:
    python train_hybrid_struct_generator_besthp.py \
      --train_csv benchmark_sequences.csv \
      --output_dir hybrid_struct_besthp_training \
      --val_fraction 0.1 \
      --modification_json rna_mod_n1methylpseudouridine_parameters.json




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




