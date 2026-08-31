# EXP029 — Observed Local Semantic BEV Sufficiency Audit

Train-only normalization/index and one final Val audit. No trajectory rollout was performed.

## Nearest-neighbor agreement

- k=1: 3-way=0.377541, binary=0.545268
- k=5: 3-way=0.422090, binary=0.547731
- k=10: 3-way=0.428249, binary=0.561486
- k=25: 3-way=0.430404, binary=0.559639

## EXP028 comparison and decision

EXP028 frozen k=25 three-way agreement was 0.444570; corrected EXP029-R1 was 0.430404. Decision: **CASE B** — coarse observed semantic BEV does not resolve the representation insufficiency identified by EXP028.

## Local consistency (k=25)

{
  "mean": 0.4998932457400945,
  "median": 0.48,
  "fraction_ge_0.8": 0.05522479983576268,
  "fraction_ge_0.9": 0.02145350030794498
}

## High-margin audit

{
  "0.25": {
    "count": 5109,
    "k25_three_way_accuracy": 0.4816989626149932,
    "k25_binary_accuracy": 0.5922881190056762,
    "probe_three_way_accuracy": 0.5044039929536113
  },
  "0.5": {
    "count": 4529,
    "k25_three_way_accuracy": 0.49966880105983663,
    "k25_binary_accuracy": 0.6056524619121219,
    "probe_three_way_accuracy": 0.5175535438286597
  },
  "1.0": {
    "count": 3821,
    "k25_three_way_accuracy": 0.5155718398325045,
    "k25_binary_accuracy": 0.613713687516357,
    "probe_three_way_accuracy": 0.527872284742214
  },
  "2.0": {
    "count": 2826,
    "k25_three_way_accuracy": 0.5481245576786978,
    "k25_binary_accuracy": 0.6422505307855626,
    "probe_three_way_accuracy": 0.5463552724699221
  }
}

## Probe

{
  "architecture": "BEV Conv(15\u219232\u219264)+GELU+AdaptiveAvgPool; base Linear\u219264; head Linear(128\u219264\u21923)",
  "epochs": 20,
  "batch_size": 256,
  "learning_rate": 0.001,
  "loss": "CrossEntropyLoss",
  "train_final_cross_entropy": 0.9309478847520488,
  "train_final_accuracy": 0.524388150894175,
  "train_history": [
    {
      "cross_entropy": 1.044489851111025,
      "accuracy": 0.4368928706278104
    },
    {
      "cross_entropy": 0.9997479506388092,
      "accuracy": 0.46679023787457524
    },
    {
      "cross_entropy": 0.9865740947897212,
      "accuracy": 0.47684756118491056
    },
    {
      "cross_entropy": 0.9795928573164837,
      "accuracy": 0.48096660144852915
    },
    {
      "cross_entropy": 0.9744414512432595,
      "accuracy": 0.4835410016132908
    },
    {
      "cross_entropy": 0.9712183128624118,
      "accuracy": 0.48560052174510004
    },
    {
      "cross_entropy": 0.9682773053574479,
      "accuracy": 0.48930765798235676
    },
    {
      "cross_entropy": 0.9652091109662224,
      "accuracy": 0.49445645831188
    },
    {
      "cross_entropy": 0.9623876583975981,
      "accuracy": 0.49455943431847044
    },
    {
      "cross_entropy": 0.9597901545678027,
      "accuracy": 0.49775169052277485
    },
    {
      "cross_entropy": 0.9570963661542936,
      "accuracy": 0.500223114680946
    },
    {
      "cross_entropy": 0.9550268539307689,
      "accuracy": 0.5022826348127553
    },
    {
      "cross_entropy": 0.951881299580923,
      "accuracy": 0.5057151683657708
    },
    {
      "cross_entropy": 0.9488821488630922,
      "accuracy": 0.5098342086293893
    },
    {
      "cross_entropy": 0.9475460057318675,
      "accuracy": 0.5083925445371229
    },
    {
      "cross_entropy": 0.9444904446528214,
      "accuracy": 0.5119280540967288
    },
    {
      "cross_entropy": 0.9405548680382277,
      "accuracy": 0.5200288332818453
    },
    {
      "cross_entropy": 0.9363785789546738,
      "accuracy": 0.5225689081110768
    },
    {
      "cross_entropy": 0.9344759908063083,
      "accuracy": 0.5223286307623657
    },
    {
      "cross_entropy": 0.9309478847520488,
      "accuracy": 0.524388150894175
    }
  ],
  "val_three_way_accuracy": 0.4774173680969,
  "val_binary_move_stay_accuracy": 0.6309792650379799,
  "val_confusion": [
    [
      3000,
      798,
      467
    ],
    [
      1476,
      1047,
      662
    ],
    [
      854,
      834,
      604
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
