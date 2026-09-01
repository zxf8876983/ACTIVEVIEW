# EXP036

Legal single-step dense, pairwise, graph and Bayesian diagnostics.

```json
{
  "experiment_id": "EXP036",
  "status": "COMPLETED",
  "split": "val",
  "test_used": false,
  "training_performed": true,
  "methods": {
    "DenseRegression": {
      "train_final_loss": 2.296612055475724,
      "model": "legal state + 9-D viewpoint descriptor"
    },
    "BradleyTerry": {
      "train_final_loss": 0.49857088727808024,
      "pair_count": 70680
    },
    "GMRF": {
      "lambda": 0.25
    },
    "BayesianMean": {
      "alpha": 1.0,
      "beta": 1.0
    },
    "BayesianLCB": {
      "beta_uncertainty": 1.0
    },
    "Thompson": {
      "samples": 20,
      "seeds": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19
      ]
    },
    "Kernel": "METHOD_E_SKIPPED_FOR_SCALE"
  },
  "p2_p3": {
    "DenseRegression": {
      "winner_accuracy": 0.5204432556362247,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5267452402538532,
        "0.5": 0.5241514360313316,
        "1.0": 0.5271007639141506,
        "2.0": 0.525
      }
    },
    "BradleyTerry": {
      "winner_accuracy": 0.5034390523500191,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5001511030522816,
        "0.5": 0.5026109660574413,
        "1.0": 0.4994543470352856,
        "2.0": 0.49605263157894736
      }
    },
    "GMRF": {
      "winner_accuracy": 0.5290408865112725,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5436687821093986,
        "0.5": 0.5414490861618799,
        "1.0": 0.5463805020007275,
        "2.0": 0.543859649122807
      }
    },
    "BayesianMean": {
      "winner_accuracy": 0.5363011081390906,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5569658507101843,
        "0.5": 0.5558093994778068,
        "1.0": 0.560931247726446,
        "2.0": 0.5631578947368421
      }
    },
    "BayesianLCB": {
      "winner_accuracy": 0.5363011081390906,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5569658507101843,
        "0.5": 0.5558093994778068,
        "1.0": 0.560931247726446,
        "2.0": 0.5631578947368421
      }
    },
    "ThompsonMean": {
      "winner_accuracy": 0.5376385173863202,
      "episode_count": 5234,
      "high_margin": {
        "0.25": 0.5587790873375642,
        "0.5": 0.5577676240208878,
        "1.0": 0.5634776282284467,
        "2.0": 0.5644736842105263
      }
    }
  },
  "full_32_view": {
    "DenseRegression": {
      "selected_true_ce": 2.047935953642801,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 1.9708348465819512,
      "top1_oracle_hit": 0.04060913705583756,
      "top3_oracle_hit": 0.10152284263959391,
      "selected_rank_mean": 15.304568527918782,
      "improvement_relative_current": -0.7077656807427344
    },
    "BradleyTerry": {
      "selected_true_ce": 10.715848781465112,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 10.638747674404263,
      "top1_oracle_hit": 0.04060913705583756,
      "top3_oracle_hit": 0.08121827411167512,
      "selected_rank_mean": 18.380710659898476,
      "improvement_relative_current": -9.375678508565045
    },
    "GMRF": {
      "selected_true_ce": 2.3953056809348663,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 2.318204573874017,
      "top1_oracle_hit": 0.03553299492385787,
      "top3_oracle_hit": 0.1116751269035533,
      "selected_rank_mean": 16.101522842639593,
      "improvement_relative_current": -1.0551354080347999
    },
    "BayesianMean": {
      "selected_true_ce": 2.2669846291205733,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 2.1898835220597235,
      "top1_oracle_hit": 0.025380710659898477,
      "top3_oracle_hit": 0.1116751269035533,
      "selected_rank_mean": 15.355329949238579,
      "improvement_relative_current": -0.9268143562205066
    },
    "BayesianLCB": {
      "selected_true_ce": 2.2669846291205733,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 2.1898835220597235,
      "top1_oracle_hit": 0.025380710659898477,
      "top3_oracle_hit": 0.1116751269035533,
      "selected_rank_mean": 15.355329949238579,
      "improvement_relative_current": -0.9268143562205066
    },
    "ThompsonMean": {
      "selected_true_ce": 2.2669846291205733,
      "best_possible_ce": 0.07710110706084967,
      "ce_regret": 2.1898835220597235,
      "top1_oracle_hit": 0.025380710659898477,
      "top3_oracle_hit": 0.1116751269035533,
      "selected_rank_mean": 15.355329949238579,
      "improvement_relative_current": -0.9268143562205066
    }
  },
  "dense_supervision_record_count": 589,
  "legal_future_input": false,
  "leakage_flags": {
    "future_candidate_skeleton_used_at_inference": false,
    "future_true_ce_used_at_inference": false,
    "test_used": false
  },
  "provenance": {
    "source_commit": "46e0da21e9ced1004be50b9bb509d02f30450bc0",
    "stage_b_summary_sha256": "3cb52e01e1a36de6ec580c075d39345d01737820fe7313a4e6fe8312136b295f"
  }
}
```
