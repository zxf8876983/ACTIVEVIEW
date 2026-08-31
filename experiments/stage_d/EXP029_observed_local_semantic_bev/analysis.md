# EXP029 — Observed Local Semantic BEV Sufficiency Audit

Train-only normalization/index and one final Val audit. No trajectory rollout was performed.

## Nearest-neighbor agreement

- k=1: 3-way=0.373229, binary=0.547013
- k=5: 3-way=0.421166, binary=0.552761
- k=10: 3-way=0.422808, binary=0.557072
- k=25: 3-way=0.426709, binary=0.559844

## Probe

{
  "architecture": "BEV Conv(15\u219232\u219264)+GELU+AdaptiveAvgPool; base Linear\u219264; head Linear(128\u219264\u21923)",
  "epochs": 20,
  "batch_size": 256,
  "learning_rate": 0.001,
  "loss": "CrossEntropyLoss",
  "train_final_cross_entropy": 0.9310285369624718,
  "train_final_accuracy": 0.5240105722033432,
  "train_history": [
    {
      "cross_entropy": 1.0444778905111103,
      "accuracy": 0.43675556928568976
    },
    {
      "cross_entropy": 0.9998205650115862,
      "accuracy": 0.46661861119692444
    },
    {
      "cross_entropy": 0.986614185575967,
      "accuracy": 0.47701918786256137
    },
    {
      "cross_entropy": 0.9796193742573669,
      "accuracy": 0.4817217588301926
    },
    {
      "cross_entropy": 0.9744600154702688,
      "accuracy": 0.48309477225139874
    },
    {
      "cross_entropy": 0.9712216541024277,
      "accuracy": 0.48649298046888406
    },
    {
      "cross_entropy": 0.9683062201408346,
      "accuracy": 0.48930765798235676
    },
    {
      "cross_entropy": 0.965234966954958,
      "accuracy": 0.49390725294339755
    },
    {
      "cross_entropy": 0.9624367113200643,
      "accuracy": 0.4946967356605911
    },
    {
      "cross_entropy": 0.9598468552267367,
      "accuracy": 0.49792331720042565
    },
    {
      "cross_entropy": 0.9571784362874644,
      "accuracy": 0.5000514880032952
    },
    {
      "cross_entropy": 0.9550779868697701,
      "accuracy": 0.5022139841416949
    },
    {
      "cross_entropy": 0.9519657437847109,
      "accuracy": 0.5054405656815295
    },
    {
      "cross_entropy": 0.9489532911936671,
      "accuracy": 0.5105207153399924
    },
    {
      "cross_entropy": 0.9476310753271557,
      "accuracy": 0.5080492911818213
    },
    {
      "cross_entropy": 0.9445683316541529,
      "accuracy": 0.5118250780901383
    },
    {
      "cross_entropy": 0.9406447566034734,
      "accuracy": 0.5196169292554834
    },
    {
      "cross_entropy": 0.9364635566254541,
      "accuracy": 0.5229808121374386
    },
    {
      "cross_entropy": 0.9345599393768004,
      "accuracy": 0.522191329420245
    },
    {
      "cross_entropy": 0.9310285369624718,
      "accuracy": 0.5240105722033432
    }
  ],
  "val_three_way_accuracy": 0.47772531307739685,
  "val_binary_move_stay_accuracy": 0.6313898583453089,
  "val_confusion": [
    [
      3001,
      791,
      473
    ],
    [
      1473,
      1044,
      668
    ],
    [
      854,
      829,
      609
    ]
  ]
}

## Leakage flags

- future_candidate_rgb_used=false
- future_candidate_semantic_used=false
- future_candidate_depth_used=false
- future_candidate_skeleton_used=false
- full_unobserved_semantic_map_used=false
- true_utility_used_as_model_input=false
- val_used_for_normalization=false
- val_used_for_neighbor_index=false
- test_used=false
