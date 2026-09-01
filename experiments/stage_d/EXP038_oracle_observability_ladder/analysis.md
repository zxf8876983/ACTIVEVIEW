# EXP038 — Privileged Oracle Observability Ladder

Train/Val-only analysis. Test was not read.

```json
{
  "experiment_id": "EXP038",
  "status": "COMPLETED",
  "split": [
    "train",
    "val"
  ],
  "test_used": false,
  "training_performed": true,
  "variants": {
    "L0_LEGAL": {
      "train_final_loss": 2.231095659340741,
      "candidate_metrics": {
        "n": 28390,
        "mae": 2.600373133087986,
        "rmse": 4.07402223169576,
        "pearson": 0.4744463618369393,
        "spearman": 0.5087889506248553
      },
      "winner": {
        "winner_accuracy": 0.58467125382263,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.6097339782345829
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.6161879895561357
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6290498725882782
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6346491228070176
          }
        }
      },
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.6564667190963037,
          "macro_f1": 0.6090646385600732,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.8070422535211268,
              "f1": 0.7670682730923696
            },
            "1": {
              "support": 142,
              "accuracy": 0.5,
              "f1": 0.5843621399176955
            },
            "2": {
              "support": 213,
              "accuracy": 0.4835680751173709,
              "f1": 0.5073891625615763
            },
            "3": {
              "support": 284,
              "accuracy": 0.15140845070422534,
              "f1": 0.19953596287703013
            },
            "4": {
              "support": 284,
              "accuracy": 0.6830985915492958,
              "f1": 0.7791164658634537
            },
            "5": {
              "support": 994,
              "accuracy": 0.8501006036217303,
              "f1": 0.7688808007279344
            },
            "6": {
              "support": 852,
              "accuracy": 0.5845070422535211,
              "f1": 0.6021765417170496
            },
            "7": {
              "support": 1420,
              "accuracy": 0.7985915492957747,
              "f1": 0.8283418553688823
            },
            "8": {
              "support": 1420,
              "accuracy": 0.6211267605633802,
              "f1": 0.5945399393326593
            },
            "9": {
              "support": 1207,
              "accuracy": 0.7067108533554267,
              "f1": 0.685140562248996
            },
            "10": {
              "support": 1420,
              "accuracy": 0.7894366197183098,
              "f1": 0.8027210884353742
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5274647887323943,
              "f1": 0.5246935201401051
            },
            "12": {
              "support": 1420,
              "accuracy": 0.6366197183098592,
              "f1": 0.5849239728243287
            },
            "13": {
              "support": 639,
              "accuracy": 0.5727699530516432,
              "f1": 0.6064623032311516
            },
            "14": {
              "support": 213,
              "accuracy": 0.784037558685446,
              "f1": 0.6802443991853361
            },
            "15": {
              "support": 639,
              "accuracy": 0.16588419405320814,
              "f1": 0.22943722943722944
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.4198665681889906,
          "median": 0.00380507963745913,
          "p90": 5.411194438673555
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -262.75147504049414,
          "clipped_mean": 0.559578417288611,
          "aggregate_positive_clipped_ratio": 0.7820306997699239
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.4443411739472367,
          "move_2_rate": 0.25216272252806177,
          "average_moves": 0.9486666190033602,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.61739890892843,
            "median": 2.210545301437378,
            "p90": 6.050670146942139
          }
        }
      },
      "privileged": false
    },
    "L1_GT_LABEL": {
      "train_final_loss": 1.4008407986396305,
      "candidate_metrics": {
        "n": 28390,
        "mae": 1.7594404180650522,
        "rmse": 3.147545928469532,
        "pearson": 0.7199580147836028,
        "spearman": 0.7041565801879949
      },
      "winner": {
        "winner_accuracy": 0.6022553516819572,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.6414752116082225
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.6481723237597912
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6585365853658537
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6719298245614035
          }
        }
      },
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.7025809680417531,
          "macro_f1": 0.643642091990005,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.8338028169014085,
              "f1": 0.7895965321773926
            },
            "1": {
              "support": 142,
              "accuracy": 0.5140845070422535,
              "f1": 0.5959183673469387
            },
            "2": {
              "support": 213,
              "accuracy": 0.48826291079812206,
              "f1": 0.5161290322580645
            },
            "3": {
              "support": 284,
              "accuracy": 0.15492957746478872,
              "f1": 0.20370370370370372
            },
            "4": {
              "support": 284,
              "accuracy": 0.7112676056338029,
              "f1": 0.8015873015873017
            },
            "5": {
              "support": 994,
              "accuracy": 0.8933601609657947,
              "f1": 0.7928571428571428
            },
            "6": {
              "support": 852,
              "accuracy": 0.6678403755868545,
              "f1": 0.6773809523809524
            },
            "7": {
              "support": 1420,
              "accuracy": 0.8373239436619718,
              "f1": 0.8532472192321494
            },
            "8": {
              "support": 1420,
              "accuracy": 0.6774647887323944,
              "f1": 0.6788990825688073
            },
            "9": {
              "support": 1207,
              "accuracy": 0.7597348798674399,
              "f1": 0.733013589128697
            },
            "10": {
              "support": 1420,
              "accuracy": 0.8049295774647888,
              "f1": 0.8225980568549838
            },
            "11": {
              "support": 1420,
              "accuracy": 0.6007042253521127,
              "f1": 0.5973389355742297
            },
            "12": {
              "support": 1420,
              "accuracy": 0.7133802816901409,
              "f1": 0.6563006154842891
            },
            "13": {
              "support": 639,
              "accuracy": 0.6103286384976526,
              "f1": 0.6532663316582915
            },
            "14": {
              "support": 213,
              "accuracy": 0.8262910798122066,
              "f1": 0.6591760299625468
            },
            "15": {
              "support": 639,
              "accuracy": 0.18779342723004694,
              "f1": 0.267260579064588
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.1391961705548819,
          "median": 0.001719641062663868,
          "p90": 4.274045725166798
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -1020.681805056007,
          "clipped_mean": 0.5982161329882444,
          "aggregate_positive_clipped_ratio": 0.822649151412853
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.3754915278472868,
          "move_2_rate": 0.32101236862801175,
          "average_moves": 1.0175162651033103,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.7117359542821964,
            "median": 2.35710072517395,
            "p90": 6.232566833496094
          }
        }
      },
      "privileged": true,
      "gt_activity_label_used_at_inference": true
    },
    "L2_GT_MOTION_STATE": {
      "status": "GT_MOTION_STATE_BLOCKED",
      "reason": "No canonical GT-to-source joint/coordinate mapping is exposed; the experiment does not guess a mapping."
    },
    "L3_GT_LABEL_MOTION": {
      "status": "BLOCKED_DEPENDS_ON_L2"
    },
    "CLASS_VIEW_PRIOR_ORACLE_LABEL": {
      "winner": {
        "winner_accuracy": 0.5569571865443425,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.5804111245465539
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.5842036553524804
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.5915544230069166
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6043859649122807
          }
        }
      },
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.6427396868520769,
          "macro_f1": 0.5900297899753515,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.8007042253521127,
              "f1": 0.7700643413477818
            },
            "1": {
              "support": 142,
              "accuracy": 0.4788732394366197,
              "f1": 0.5666666666666667
            },
            "2": {
              "support": 213,
              "accuracy": 0.49295774647887325,
              "f1": 0.5121951219512194
            },
            "3": {
              "support": 284,
              "accuracy": 0.11971830985915492,
              "f1": 0.1559633027522936
            },
            "4": {
              "support": 284,
              "accuracy": 0.6971830985915493,
              "f1": 0.7857142857142856
            },
            "5": {
              "support": 994,
              "accuracy": 0.869215291750503,
              "f1": 0.7773279352226721
            },
            "6": {
              "support": 852,
              "accuracy": 0.596244131455399,
              "f1": 0.6033254156769596
            },
            "7": {
              "support": 1420,
              "accuracy": 0.7605633802816901,
              "f1": 0.7964601769911505
            },
            "8": {
              "support": 1420,
              "accuracy": 0.5711267605633803,
              "f1": 0.5817790530846485
            },
            "9": {
              "support": 1207,
              "accuracy": 0.6884838442419221,
              "f1": 0.6726021853500607
            },
            "10": {
              "support": 1420,
              "accuracy": 0.7535211267605634,
              "f1": 0.7767695099818512
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5338028169014084,
              "f1": 0.5252945252945254
            },
            "12": {
              "support": 1420,
              "accuracy": 0.6169014084507042,
              "f1": 0.5647969052224371
            },
            "13": {
              "support": 639,
              "accuracy": 0.5790297339593115,
              "f1": 0.5953338696701529
            },
            "14": {
              "support": 213,
              "accuracy": 0.7981220657276995,
              "f1": 0.5128205128205128
            },
            "15": {
              "support": 639,
              "accuracy": 0.17214397496087636,
              "f1": 0.2433628318584071
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.5774186744804328,
          "median": 0.006569219287484884,
          "p90": 5.942214524839071
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -1026.112953933305,
          "clipped_mean": 0.5438829858490458,
          "aggregate_positive_clipped_ratio": 0.7570904185808686
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.3311646528919711,
          "move_2_rate": 0.36533924358332737,
          "average_moves": 1.061843140058626,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.8501542207531996,
            "median": 2.521422863006592,
            "p90": 6.6638698101043685
          }
        }
      },
      "privileged": true
    }
  },
  "observability_gap": {
    "label_delta_accuracy": null,
    "motion_delta_accuracy": null,
    "joint_delta_accuracy": null,
    "motion_status": "GT_MOTION_STATE_BLOCKED"
  },
  "leakage_flags": {
    "gt_label_variant_privileged": true,
    "gt_motion_variant_privileged": true,
    "future_candidate_quality_used_as_input": false,
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
