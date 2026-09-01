# EXP040 — Sequential Belief-Space Active HAR

Train/Val-only analysis. Test was not read.

```json
{
  "experiment_id": "EXP040",
  "status": "COMPLETED",
  "split": "val",
  "test_used": false,
  "training_performed": false,
  "state_scoring": "candidate heads use the initial legal Stage-D state; beliefs update only after visited transitions",
  "methods": {
    "Stay": {
      "1": {
        "terminal_har_accuracy": 0.6491027382569529,
        "terminal_har_macro_f1": 0.5980486801421372,
        "fused_har_accuracy": 0.6491027382569529,
        "fused_har_macro_f1": 0.5980486801421372,
        "average_moves": 0.0,
        "move_rate": 0.0,
        "path_length": 0.0,
        "terminal_true_ce": 2.2163263016440027,
        "best_visited_true_ce": 2.2163263016440027,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6491027382569529,
        "terminal_har_macro_f1": 0.5980486801421372,
        "fused_har_accuracy": 0.6491027382569529,
        "fused_har_macro_f1": 0.5980486801421372,
        "average_moves": 0.0,
        "move_rate": 0.0,
        "path_length": 0.0,
        "terminal_true_ce": 2.2163263016440027,
        "best_visited_true_ce": 2.2163263016440027,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6491027382569529,
        "terminal_har_macro_f1": 0.5980486801421372,
        "fused_har_accuracy": 0.6491027382569529,
        "fused_har_macro_f1": 0.5980486801421372,
        "average_moves": 0.0,
        "move_rate": 0.0,
        "path_length": 0.0,
        "terminal_true_ce": 2.2163263016440027,
        "best_visited_true_ce": 2.2163263016440027,
        "har_episode_count": 13987
      }
    },
    "Random": {
      "1": {
        "terminal_har_accuracy": 0.5855437191677987,
        "terminal_har_macro_f1": 0.5172133750140391,
        "fused_har_accuracy": 0.6485307785801101,
        "fused_har_macro_f1": 0.5815689248934434,
        "average_moves": 0.7561075754465202,
        "move_rate": 0.7561075754465202,
        "path_length": 0.6850776917707795,
        "terminal_true_ce": 3.4504442241647815,
        "best_visited_true_ce": 1.6063697191570792,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.5674554943876456,
        "terminal_har_macro_f1": 0.5043491530756083,
        "fused_har_accuracy": 0.6442410810037892,
        "fused_har_macro_f1": 0.5778185556830164,
        "average_moves": 1.3379182919318415,
        "move_rate": 0.7539519605830425,
        "path_length": 1.2452460638937843,
        "terminal_true_ce": 3.8980407369488317,
        "best_visited_true_ce": 1.4937467132271631,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.5576606849217131,
        "terminal_har_macro_f1": 0.49674326897809684,
        "fused_har_accuracy": 0.6393079287910203,
        "fused_har_macro_f1": 0.571837292253405,
        "average_moves": 1.79901457606241,
        "move_rate": 0.7550810921781975,
        "path_length": 1.6985176981982586,
        "terminal_true_ce": 4.014515292515785,
        "best_visited_true_ce": 1.4191669161887803,
        "har_episode_count": 13987
      }
    },
    "BELIEF_GREEDY_LATEST": {
      "1": {
        "terminal_har_accuracy": 0.6237220275970544,
        "terminal_har_macro_f1": 0.5601211786566449,
        "fused_har_accuracy": 0.6556087795810396,
        "fused_har_macro_f1": 0.5947691906958688,
        "average_moves": 0.5290494764935332,
        "move_rate": 0.5290494764935332,
        "path_length": 0.4213141085424357,
        "terminal_true_ce": 2.7824621561926572,
        "best_visited_true_ce": 1.8072919359400006,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6191463501823121,
        "terminal_har_macro_f1": 0.5559689699001481,
        "fused_har_accuracy": 0.6503896475298492,
        "fused_har_macro_f1": 0.5850412688655041,
        "average_moves": 0.825292547731472,
        "move_rate": 0.5290494764935332,
        "path_length": 0.7309028271609985,
        "terminal_true_ce": 3.1173967506949203,
        "best_visited_true_ce": 1.7176946063471445,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6192893401015228,
        "terminal_har_macro_f1": 0.5552257559814452,
        "fused_har_accuracy": 0.6452420104382641,
        "fused_har_macro_f1": 0.5778298883311438,
        "average_moves": 1.0452679121330322,
        "move_rate": 0.5290494764935332,
        "path_length": 0.951074184162408,
        "terminal_true_ce": 3.0953801864244626,
        "best_visited_true_ce": 1.6803521271401145,
        "har_episode_count": 13987
      }
    },
    "BELIEF_GREEDY_MEAN": {
      "1": {
        "terminal_har_accuracy": 0.6237220275970544,
        "terminal_har_macro_f1": 0.5601211786566449,
        "fused_har_accuracy": 0.6556087795810396,
        "fused_har_macro_f1": 0.5947691906958688,
        "average_moves": 0.5290494764935332,
        "move_rate": 0.5290494764935332,
        "path_length": 0.4213141085424357,
        "terminal_true_ce": 2.7824621561926572,
        "best_visited_true_ce": 1.8072919359400006,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6216486737684993,
        "terminal_har_macro_f1": 0.5571044772290459,
        "fused_har_accuracy": 0.6508186172874812,
        "fused_har_macro_f1": 0.5862409275921181,
        "average_moves": 0.7761239991788134,
        "move_rate": 0.5290494764935332,
        "path_length": 0.6718699323585869,
        "terminal_true_ce": 3.040637743062273,
        "best_visited_true_ce": 1.7302604750000103,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6258668763852149,
        "terminal_har_macro_f1": 0.5594437556754859,
        "fused_har_accuracy": 0.649746192893401,
        "fused_har_macro_f1": 0.5852775240361321,
        "average_moves": 0.9162389653048655,
        "move_rate": 0.5290494764935332,
        "path_length": 0.8137636262903595,
        "terminal_true_ce": 2.987662557916309,
        "best_visited_true_ce": 1.6958323493400027,
        "har_episode_count": 13987
      }
    },
    "BELIEF_GREEDY_GEOMETRIC": {
      "1": {
        "terminal_har_accuracy": 0.6237220275970544,
        "terminal_har_macro_f1": 0.5601211786566449,
        "fused_har_accuracy": 0.6556087795810396,
        "fused_har_macro_f1": 0.5947691906958688,
        "average_moves": 0.5290494764935332,
        "move_rate": 0.5290494764935332,
        "path_length": 0.4213141085424357,
        "terminal_true_ce": 2.7824621561926572,
        "best_visited_true_ce": 1.8072919359400006,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6182169157074426,
        "terminal_har_macro_f1": 0.5550682765271386,
        "fused_har_accuracy": 0.6533209408736684,
        "fused_har_macro_f1": 0.5893960153207958,
        "average_moves": 0.7512831040854034,
        "move_rate": 0.5290494764935332,
        "path_length": 0.6317424028160655,
        "terminal_true_ce": 2.910605910175055,
        "best_visited_true_ce": 1.7426300158300336,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6197183098591549,
        "terminal_har_macro_f1": 0.5554076249206683,
        "fused_har_accuracy": 0.6525344963180095,
        "fused_har_macro_f1": 0.5878512003868658,
        "average_moves": 0.8664545267912133,
        "move_rate": 0.5290494764935332,
        "path_length": 0.7449533527867467,
        "terminal_true_ce": 2.8812066349358574,
        "best_visited_true_ce": 1.7069855427459206,
        "har_episode_count": 13987
      }
    },
    "BELIEF_CORRECTNESS_GREEDY": {
      "1": {
        "terminal_har_accuracy": 0.6535354257524845,
        "terminal_har_macro_f1": 0.5888358852519304,
        "fused_har_accuracy": 0.6583255880460428,
        "fused_har_macro_f1": 0.5961209464244519,
        "average_moves": 0.2261342640114966,
        "move_rate": 0.2261342640114966,
        "path_length": 0.1776096307550385,
        "terminal_true_ce": 2.573750759274774,
        "best_visited_true_ce": 2.03305290056009,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6495317080145849,
        "terminal_har_macro_f1": 0.5815337904622018,
        "fused_har_accuracy": 0.6611138914706514,
        "fused_har_macro_f1": 0.5970331021030882,
        "average_moves": 0.3660439334838842,
        "move_rate": 0.2261342640114966,
        "path_length": 0.3234744512492661,
        "terminal_true_ce": 2.7809425069182803,
        "best_visited_true_ce": 1.9902306912318402,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6499606777722171,
        "terminal_har_macro_f1": 0.5812313271541678,
        "fused_har_accuracy": 0.6586115678844642,
        "fused_har_macro_f1": 0.5927297946749251,
        "average_moves": 0.47885444467255184,
        "move_rate": 0.2261342640114966,
        "path_length": 0.4317410899746075,
        "terminal_true_ce": 2.78735623291503,
        "best_visited_true_ce": 1.973875295457581,
        "har_episode_count": 13987
      }
    },
    "GT_LABEL_BELIEF_UPPER_BOUND": {
      "1": {
        "terminal_har_accuracy": 0.6653320940873668,
        "terminal_har_macro_f1": 0.5994903091391961,
        "fused_har_accuracy": 0.6822049045542289,
        "fused_har_macro_f1": 0.619760798140673,
        "average_moves": 0.6758365838636831,
        "move_rate": 0.6758365838636831,
        "path_length": 0.5957711913476267,
        "terminal_true_ce": 2.554940742841081,
        "best_visited_true_ce": 1.3517948798527137,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.6593265174805176,
        "terminal_har_macro_f1": 0.5942885869683237,
        "fused_har_accuracy": 0.6897118753127904,
        "fused_har_macro_f1": 0.6261423227630587,
        "average_moves": 0.9948675836583863,
        "move_rate": 0.6758365838636831,
        "path_length": 0.9141052551488208,
        "terminal_true_ce": 2.805855046409459,
        "best_visited_true_ce": 1.1141583645186044,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.6613998713090727,
        "terminal_har_macro_f1": 0.5961165592636457,
        "fused_har_accuracy": 0.6938585829699007,
        "fused_har_macro_f1": 0.6281948580063599,
        "average_moves": 1.1470950523506467,
        "move_rate": 0.6758365838636831,
        "path_length": 1.0686516306463234,
        "terminal_true_ce": 2.8306011069771464,
        "best_visited_true_ce": 1.0117134166030222,
        "har_episode_count": 13987
      }
    },
    "TRUE_CE_GRAPH_ORACLE": {
      "1": {
        "terminal_har_accuracy": 0.7991706584685779,
        "terminal_har_macro_f1": 0.7510724962458131,
        "fused_har_accuracy": 0.7577035818974762,
        "fused_har_macro_f1": 0.709875278926291,
        "average_moves": 0.7293163621432971,
        "move_rate": 0.7293163621432971,
        "path_length": 0.6244157471238428,
        "terminal_true_ce": 0.7595879479243772,
        "best_visited_true_ce": 0.7595879479243772,
        "har_episode_count": 13987
      },
      "2": {
        "terminal_har_accuracy": 0.8229069850575534,
        "terminal_har_macro_f1": 0.7818977786200129,
        "fused_har_accuracy": 0.7828698076785586,
        "fused_har_macro_f1": 0.7352997782218629,
        "average_moves": 1.0042085814001231,
        "move_rate": 0.7293163621432971,
        "path_length": 0.8582142182378516,
        "terminal_true_ce": 0.5815603726595953,
        "best_visited_true_ce": 0.5815603726595953,
        "har_episode_count": 13987
      },
      "3": {
        "terminal_har_accuracy": 0.8271966826338744,
        "terminal_har_macro_f1": 0.7872002810623635,
        "fused_har_accuracy": 0.7872310002144849,
        "fused_har_macro_f1": 0.7394457324410331,
        "average_moves": 1.0748306302607267,
        "move_rate": 0.7293163621432971,
        "path_length": 0.9166794170574267,
        "terminal_true_ce": 0.5475608879379344,
        "best_visited_true_ce": 0.5475608879379344,
        "har_episode_count": 13987
      }
    }
  },
  "belief_update_rules": {
    "latest": "last visited q",
    "mean": "arithmetic mean then normalize",
    "geometric": "mean log posterior, eps=1e-8"
  },
  "leakage_flags": {
    "unvisited_viewpoint_output_used": false,
    "planner_reads_only_selected_visited_belief": true,
    "belief_cache_materialized_before_rollout": true,
    "gt_label_planner_input": true,
    "true_ce_planner_input_for_privileged_oracle": true,
    "true_ce_evaluator_only_for_legal_methods": true,
    "test_used": false
  },
  "provenance": {
    "source_commit": "6d0f97ed39f98abb277b8ecd9c606dbc934a4fce",
    "stage_d_feature_summary_sha256": "5ada9ca595aca421b77c507ee6ea3ccd19eda0008fda943f4483e5a41837adfb",
    "dense_field_root": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field",
    "test_used": false
  }
}
```
