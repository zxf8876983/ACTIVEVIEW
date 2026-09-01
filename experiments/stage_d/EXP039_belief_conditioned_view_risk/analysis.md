# EXP039 — Deployable Belief-Conditioned View Risk

Train/Val-only analysis. Test was not read.

```json
{
  "experiment_id": "EXP039",
  "status": "COMPLETED",
  "split": "val",
  "test_used": false,
  "training_performed": true,
  "model": {
    "ce_head": {
      "final_loss": 1.400275301179758,
      "masked_target": true
    },
    "correctness_head": {
      "final_loss": 0.3474631950168424,
      "masked_target": true
    }
  },
  "current_belief_audit": {
    "count": 9742,
    "top1_accuracy": 0.6089098747690412,
    "top2_coverage": 0.7522069390268938,
    "top3_coverage": 0.8259084376924656,
    "top5_coverage": 0.9076165058509547,
    "mean_entropy": 0.3592353620865182
  },
  "methods": {
    "MAP_CLASS_RISK": {
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.6468148995495817,
          "macro_f1": 0.5956047207246984,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.773943661971831,
              "f1": 0.7639902676399026
            },
            "1": {
              "support": 142,
              "accuracy": 0.43661971830985913,
              "f1": 0.5188284518828452
            },
            "2": {
              "support": 213,
              "accuracy": 0.4647887323943662,
              "f1": 0.48410757946210264
            },
            "3": {
              "support": 284,
              "accuracy": 0.15492957746478872,
              "f1": 0.1900647948164147
            },
            "4": {
              "support": 284,
              "accuracy": 0.6654929577464789,
              "f1": 0.7636363636363637
            },
            "5": {
              "support": 994,
              "accuracy": 0.8350100603621731,
              "f1": 0.758337140246688
            },
            "6": {
              "support": 852,
              "accuracy": 0.5950704225352113,
              "f1": 0.5919439579684763
            },
            "7": {
              "support": 1420,
              "accuracy": 0.780281690140845,
              "f1": 0.8162062615101289
            },
            "8": {
              "support": 1420,
              "accuracy": 0.6190140845070422,
              "f1": 0.6012311901504787
            },
            "9": {
              "support": 1207,
              "accuracy": 0.6992543496271748,
              "f1": 0.6864579097193981
            },
            "10": {
              "support": 1420,
              "accuracy": 0.7746478873239436,
              "f1": 0.7857142857142857
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5415492957746478,
              "f1": 0.5250938886992148
            },
            "12": {
              "support": 1420,
              "accuracy": 0.6422535211267606,
              "f1": 0.5797838525111252
            },
            "13": {
              "support": 639,
              "accuracy": 0.5414710485133021,
              "f1": 0.5676784249384742
            },
            "14": {
              "support": 213,
              "accuracy": 0.7370892018779343,
              "f1": 0.6781857451403889
            },
            "15": {
              "support": 639,
              "accuracy": 0.1596244131455399,
              "f1": 0.21841541755888652
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.4946578795280923,
          "median": 0.004460936458599463,
          "p90": 5.759641271363941
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -340.6227763012324,
          "clipped_mean": 0.5506420040203248,
          "aggregate_positive_clipped_ratio": 0.7728182543210146
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.5135482948452134,
          "move_2_rate": 0.1829556016300851,
          "average_moves": 0.8794594981053836,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.4379323649638254,
            "median": 2.124786615371704,
            "p90": 5.318538498878479
          }
        }
      },
      "winner": {
        "winner_accuracy": 0.5726299694189603,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.5922007255139057
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.5972584856396866
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6053876956680014
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6105263157894737
          }
        }
      }
    },
    "BELIEF_EXPECTED_RISK": {
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.6478158289840567,
          "macro_f1": 0.5961541954575309,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.7852112676056338,
              "f1": 0.7657967032967032
            },
            "1": {
              "support": 142,
              "accuracy": 0.4295774647887324,
              "f1": 0.5126050420168067
            },
            "2": {
              "support": 213,
              "accuracy": 0.45539906103286387,
              "f1": 0.48019801980198024
            },
            "3": {
              "support": 284,
              "accuracy": 0.1619718309859155,
              "f1": 0.19956616052060738
            },
            "4": {
              "support": 284,
              "accuracy": 0.6619718309859155,
              "f1": 0.7595959595959596
            },
            "5": {
              "support": 994,
              "accuracy": 0.8400402414486922,
              "f1": 0.7604735883424409
            },
            "6": {
              "support": 852,
              "accuracy": 0.5927230046948356,
              "f1": 0.5920281359906213
            },
            "7": {
              "support": 1420,
              "accuracy": 0.7809859154929577,
              "f1": 0.8178466076696165
            },
            "8": {
              "support": 1420,
              "accuracy": 0.6211267605633802,
              "f1": 0.6020477815699659
            },
            "9": {
              "support": 1207,
              "accuracy": 0.6934548467274234,
              "f1": 0.6860655737704918
            },
            "10": {
              "support": 1420,
              "accuracy": 0.773943661971831,
              "f1": 0.7866857551896922
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5352112676056338,
              "f1": 0.5217988328184003
            },
            "12": {
              "support": 1420,
              "accuracy": 0.647887323943662,
              "f1": 0.5817262092949731
            },
            "13": {
              "support": 639,
              "accuracy": 0.543035993740219,
              "f1": 0.5707236842105264
            },
            "14": {
              "support": 213,
              "accuracy": 0.7464788732394366,
              "f1": 0.6838709677419356
            },
            "15": {
              "support": 639,
              "accuracy": 0.15805946791862285,
              "f1": 0.21743810548977396
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.4857497248442324,
          "median": 0.004269338911399245,
          "p90": 5.706415862217544
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -346.75255313705344,
          "clipped_mean": 0.5517230290279683,
          "aggregate_positive_clipped_ratio": 0.7760596568707799
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.5131908200471866,
          "move_2_rate": 0.18331307642811182,
          "average_moves": 0.8798169729034103,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.433480663914741,
            "median": 2.1247870922088623,
            "p90": 5.247164201736448
          }
        }
      },
      "winner": {
        "winner_accuracy": 0.5747324159021406,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.5955259975816203
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.5998694516971279
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6101201310520568
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6157894736842106
          }
        }
      }
    },
    "TOP3_BELIEF_RISK": {
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.6479588189032673,
          "macro_f1": 0.5962464702476984,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.7845070422535211,
              "f1": 0.7661623108665749
            },
            "1": {
              "support": 142,
              "accuracy": 0.4295774647887324,
              "f1": 0.5126050420168067
            },
            "2": {
              "support": 213,
              "accuracy": 0.45539906103286387,
              "f1": 0.47901234567901235
            },
            "3": {
              "support": 284,
              "accuracy": 0.1619718309859155,
              "f1": 0.19913419913419914
            },
            "4": {
              "support": 284,
              "accuracy": 0.6654929577464789,
              "f1": 0.7620967741935484
            },
            "5": {
              "support": 994,
              "accuracy": 0.8390342052313883,
              "f1": 0.7602552415679125
            },
            "6": {
              "support": 852,
              "accuracy": 0.5938967136150235,
              "f1": 0.591812865497076
            },
            "7": {
              "support": 1420,
              "accuracy": 0.7845070422535211,
              "f1": 0.8197203826342899
            },
            "8": {
              "support": 1420,
              "accuracy": 0.6197183098591549,
              "f1": 0.6017094017094018
            },
            "9": {
              "support": 1207,
              "accuracy": 0.6934548467274234,
              "f1": 0.6860655737704918
            },
            "10": {
              "support": 1420,
              "accuracy": 0.7746478873239436,
              "f1": 0.7862759113652609
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5338028169014084,
              "f1": 0.5213204951856947
            },
            "12": {
              "support": 1420,
              "accuracy": 0.6471830985915493,
              "f1": 0.5810938981979133
            },
            "13": {
              "support": 639,
              "accuracy": 0.5446009389671361,
              "f1": 0.5723684210526315
            },
            "14": {
              "support": 213,
              "accuracy": 0.7464788732394366,
              "f1": 0.6824034334763949
            },
            "15": {
              "support": 639,
              "accuracy": 0.15805946791862285,
              "f1": 0.21790722761596545
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.4849415357032973,
          "median": 0.004297346054954687,
          "p90": 5.709115109610138
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -346.584628358421,
          "clipped_mean": 0.5516865001240377,
          "aggregate_positive_clipped_ratio": 0.7760671029882745
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.5147637091585043,
          "move_2_rate": 0.18174018731679417,
          "average_moves": 0.8782440837920926,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.431267323266056,
            "median": 2.1247870922088623,
            "p90": 5.233090019226071
          }
        }
      },
      "winner": {
        "winner_accuracy": 0.5753058103975535,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.5964328899637243
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.5998694516971279
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6101201310520568
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6157894736842106
          }
        }
      }
    },
    "BELIEF_EXPECTED_CORRECTNESS": {
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.5349967827268177,
          "macro_f1": 0.46924757727190153,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.7225352112676057,
              "f1": 0.6602316602316602
            },
            "1": {
              "support": 142,
              "accuracy": 0.38028169014084506,
              "f1": 0.437246963562753
            },
            "2": {
              "support": 213,
              "accuracy": 0.26291079812206575,
              "f1": 0.28211586901763225
            },
            "3": {
              "support": 284,
              "accuracy": 0.07394366197183098,
              "f1": 0.0909090909090909
            },
            "4": {
              "support": 284,
              "accuracy": 0.5704225352112676,
              "f1": 0.6835443037974684
            },
            "5": {
              "support": 994,
              "accuracy": 0.7193158953722334,
              "f1": 0.6541628545288196
            },
            "6": {
              "support": 852,
              "accuracy": 0.4507042253521127,
              "f1": 0.4717444717444717
            },
            "7": {
              "support": 1420,
              "accuracy": 0.7049295774647887,
              "f1": 0.7434088377274415
            },
            "8": {
              "support": 1420,
              "accuracy": 0.43802816901408453,
              "f1": 0.49680511182108633
            },
            "9": {
              "support": 1207,
              "accuracy": 0.5608947804473903,
              "f1": 0.5519771708112515
            },
            "10": {
              "support": 1420,
              "accuracy": 0.7098591549295775,
              "f1": 0.733090909090909
            },
            "11": {
              "support": 1420,
              "accuracy": 0.4154929577464789,
              "f1": 0.4196301564722618
            },
            "12": {
              "support": 1420,
              "accuracy": 0.49859154929577465,
              "f1": 0.48261758691206547
            },
            "13": {
              "support": 639,
              "accuracy": 0.37871674491392804,
              "f1": 0.4416058394160584
            },
            "14": {
              "support": 213,
              "accuracy": 0.784037558685446,
              "f1": 0.2372159090909091
            },
            "15": {
              "support": 639,
              "accuracy": 0.0782472613458529,
              "f1": 0.12165450121654502
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 2.7141020210419495,
          "median": 0.19792937277816236,
          "p90": 9.346583622694013
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -1345.7919244245106,
          "clipped_mean": 0.4200695354628944,
          "aggregate_positive_clipped_ratio": 0.5925607424501951
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.05469364409809108,
          "move_2_rate": 0.6418102523772075,
          "average_moves": 1.3383141488525059,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 3.6401831064813512,
            "median": 3.6480499505996704,
            "p90": 7.414730548858643
          }
        }
      },
      "winner": {
        "winner_accuracy": 0.4294724770642202,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.41021765417170497
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.40796344647519583
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.4026210411357845
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.4
          }
        }
      }
    },
    "GT_LABEL_HEAD_UPPER_BOUND": {
      "trajectory": {
        "episode_count": 13987,
        "recognition": {
          "n": 13987,
          "accuracy": 0.7026524630013584,
          "macro_f1": 0.644127819595786,
          "per_class": {
            "0": {
              "support": 1420,
              "accuracy": 0.8401408450704225,
              "f1": 0.7942743009320905
            },
            "1": {
              "support": 142,
              "accuracy": 0.4788732394366197,
              "f1": 0.576271186440678
            },
            "2": {
              "support": 213,
              "accuracy": 0.48826291079812206,
              "f1": 0.5187032418952618
            },
            "3": {
              "support": 284,
              "accuracy": 0.18309859154929578,
              "f1": 0.23798627002288333
            },
            "4": {
              "support": 284,
              "accuracy": 0.704225352112676,
              "f1": 0.7984031936127743
            },
            "5": {
              "support": 994,
              "accuracy": 0.8843058350100603,
              "f1": 0.7918918918918919
            },
            "6": {
              "support": 852,
              "accuracy": 0.6666666666666666,
              "f1": 0.6733847065797273
            },
            "7": {
              "support": 1420,
              "accuracy": 0.8225352112676056,
              "f1": 0.847297787450127
            },
            "8": {
              "support": 1420,
              "accuracy": 0.680281690140845,
              "f1": 0.6807610993657506
            },
            "9": {
              "support": 1207,
              "accuracy": 0.7630488815244407,
              "f1": 0.7382765531062123
            },
            "10": {
              "support": 1420,
              "accuracy": 0.8,
              "f1": 0.8149210903873746
            },
            "11": {
              "support": 1420,
              "accuracy": 0.5957746478873239,
              "f1": 0.5953553835327234
            },
            "12": {
              "support": 1420,
              "accuracy": 0.726056338028169,
              "f1": 0.6630225080385852
            },
            "13": {
              "support": 639,
              "accuracy": 0.6244131455399061,
              "f1": 0.6611433305716653
            },
            "14": {
              "support": 213,
              "accuracy": 0.8544600938967136,
              "f1": 0.6582278481012658
            },
            "15": {
              "support": 639,
              "accuracy": 0.17996870109546165,
              "f1": 0.2561247216035635
            }
          }
        },
        "decision_regret": {
          "count": 13987,
          "mean": 1.112423741149743,
          "median": 0.0013918843469582498,
          "p90": 4.198239314310558
        },
        "positive_headroom_capture": {
          "episode_count": 11879,
          "raw_mean": -184.23019383300232,
          "clipped_mean": 0.6017303033343073,
          "aggregate_positive_clipped_ratio": 0.8245692018595452
        },
        "movement": {
          "move_0_rate": 0.3034961035247015,
          "move_1_rate": 0.3766354472009723,
          "move_2_rate": 0.31986844927432617,
          "average_moves": 1.0163723457496248,
          "trajectory_geodesic_cost_m": {
            "count": 13987,
            "mean": 2.712846653283073,
            "median": 2.35710072517395,
            "p90": 6.233509063720702
          }
        }
      },
      "winner": {
        "winner_accuracy": 0.6135321100917431,
        "episode_count": 5232,
        "high_margin": {
          "0.25": {
            "count": 3308,
            "accuracy": 0.6499395405078597
          },
          "0.5": {
            "count": 3064,
            "accuracy": 0.6586161879895561
          },
          "1.0": {
            "count": 2747,
            "accuracy": 0.6709137240626137
          },
          "2.0": {
            "count": 2280,
            "accuracy": 0.6833333333333333
          }
        }
      }
    }
  },
  "gt_label_head_upper_bound": true,
  "leakage_flags": {
    "gt_label_used_in_legal_inference": false,
    "gt_motion_used_in_inference": false,
    "true_ce_evaluator_only": true,
    "future_candidate_perception_used_as_input": false,
    "test_used": false
  },
  "candidate_ce_metrics_for_true_head": {
    "n": 28390,
    "mae": 1.796154704316512,
    "rmse": 3.1538466507982124,
    "pearson": 0.7182032193361548,
    "spearman": 0.6995443912542894
  },
  "provenance": {
    "source_commit": "6d0f97ed39f98abb277b8ecd9c606dbc934a4fce",
    "stage_d_feature_summary_sha256": "5ada9ca595aca421b77c507ee6ea3ccd19eda0008fda943f4483e5a41837adfb",
    "dense_field_root": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field",
    "test_used": false
  }
}
```
