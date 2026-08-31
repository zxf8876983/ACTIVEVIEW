# EXP031 — Human Viewpoint Sensitivity / Perception-Quality Audit

{
  "experiment_id": "EXP031",
  "status": "COMPLETED",
  "split": "val",
  "test_used": false,
  "training_performed": true,
  "perception_regenerated": false,
  "habitat_rendering_performed": false,
  "stgcn_retrained": false,
  "eligible_episode_counts": {
    "train": 29133,
    "val": 9742
  },
  "method_a_body_view_geometry": {
    "feature_names": [
      "candidate_human_distance",
      "candidate_human_azimuth",
      "candidate_human_elevation",
      "camera_forward_root_cosine",
      "shoulder_plane_view_angle",
      "hip_plane_view_angle",
      "body_facing_cosine",
      "front_view",
      "back_view",
      "left_side_view",
      "right_side_view",
      "projected_bbox_width",
      "projected_bbox_height",
      "projected_bbox_area",
      "projected_aspect_ratio",
      "mean_projected_bone_length",
      "min_projected_bone_length",
      "foreshortening_mean",
      "pairwise_projected_overlap",
      "left_right_limb_overlap",
      "upper_lower_overlap",
      "joint_inframe_fraction",
      "image_center_offset"
    ],
    "feature_dim": 23,
    "oracle_human_geometry_used": true,
    "human_geometry_source": "frame-15 MotionConverter kinematic anchors transformed by canonical placement/root rotation; no Habitat rendering",
    "oracle_move_episode_count": 5234,
    "feature_correlations": {
      "candidate_human_distance": {
        "pearson": -0.2037069444908636,
        "spearman": -0.1556776754957303
      },
      "candidate_human_azimuth": {
        "pearson": -0.006674975898725375,
        "spearman": -0.01274410643103515
      },
      "candidate_human_elevation": {
        "pearson": -0.1756536847799228,
        "spearman": -0.15432344384274108
      },
      "camera_forward_root_cosine": {
        "pearson": 0.12968650175159283,
        "spearman": 0.135250080118977
      },
      "shoulder_plane_view_angle": {
        "pearson": 0.01003642056448959,
        "spearman": 0.013210249959046398
      },
      "hip_plane_view_angle": {
        "pearson": -0.01799341597571778,
        "spearman": -0.015958841590928866
      },
      "body_facing_cosine": {
        "pearson": -0.03684582122220185,
        "spearman": -0.040790535058834065
      },
      "front_view": {
        "pearson": -0.03541980390269179,
        "spearman": -0.029960759101790038
      },
      "back_view": {
        "pearson": 0.0323458799853853,
        "spearman": 0.04734897191985998
      },
      "left_side_view": {
        "pearson": 0.0012633901823367807,
        "spearman": -0.001874564379605567
      },
      "right_side_view": {
        "pearson": -0.03701471260445915,
        "spearman": -0.022158932196320688
      },
      "projected_bbox_width": {
        "pearson": 0.11360967367934162,
        "spearman": 0.11796754208718516
      },
      "projected_bbox_height": {
        "pearson": 0.1228330282801617,
        "spearman": 0.1069874566395119
      },
      "projected_bbox_area": {
        "pearson": 0.12508511395317434,
        "spearman": 0.1304824384558585
      },
      "projected_aspect_ratio": {
        "pearson": 0.03528901279207589,
        "spearman": 0.027675812581195603
      },
      "mean_projected_bone_length": {
        "pearson": 0.12201980561161373,
        "spearman": 0.13116857934494117
      },
      "min_projected_bone_length": {
        "pearson": 0.10065577297503263,
        "spearman": 0.09377112340227836
      },
      "foreshortening_mean": {
        "pearson": -0.16484873713235224,
        "spearman": -0.13276294686962603
      },
      "pairwise_projected_overlap": {
        "pearson": -0.17272943476016514,
        "spearman": -0.1331528852688168
      },
      "left_right_limb_overlap": {
        "pearson": -0.14179366550440256,
        "spearman": -0.09695996558150403
      },
      "upper_lower_overlap": {
        "pearson": -0.12304482275250175,
        "spearman": -0.06798981571547202
      },
      "joint_inframe_fraction": {
        "pearson": -0.09984182183559422,
        "spearman": -0.10256607218672148
      },
      "image_center_offset": {
        "pearson": 0.1289863442355197,
        "spearman": 0.1166142121084788
      }
    },
    "analytic_winner_feature": "candidate_human_distance",
    "winner_accuracy": 0.4531906763469622,
    "high_margin": {
      "0.25": {
        "count": 2415,
        "analytic_feature": "candidate_human_distance",
        "three_way_accuracy": 0.41283643892339544
      },
      "0.5": {
        "count": 2080,
        "analytic_feature": "candidate_human_distance",
        "three_way_accuracy": 0.40721153846153846
      },
      "1.0": {
        "count": 1674,
        "analytic_feature": "candidate_human_distance",
        "three_way_accuracy": 0.3942652329749104
      },
      "2.0": {
        "count": 1145,
        "analytic_feature": "candidate_human_distance",
        "three_way_accuracy": 0.3746724890829694
      }
    },
    "future_candidate_rgb_used": false,
    "future_candidate_depth_used": false,
    "future_candidate_semantic_used": false,
    "future_candidate_skeleton_used": false,
    "test_used": false
  },
  "method_b_pose_quality": {
    "available": true,
    "feature_names": [
      "mean_pose_confidence",
      "valid_joint_fraction",
      "skeleton_spread",
      "temporal_motion_energy"
    ],
    "feature_correlations": {
      "mean_pose_confidence": {
        "pearson": 0.41806674246217534,
        "spearman": 0.23603830751651714
      },
      "valid_joint_fraction": {
        "pearson": null,
        "spearman": null
      },
      "skeleton_spread": {
        "pearson": -0.12313161597087768,
        "spearman": -0.07382960086551227
      },
      "temporal_motion_energy": {
        "pearson": -0.013963133131263385,
        "spearman": 0.05944007354620461
      }
    },
    "oracle_move_winner_accuracy": 0.5603744745892243,
    "oracle_move_winner_accuracy_by_feature": {
      "mean_pose_confidence": 0.5783339701948796,
      "valid_joint_fraction": 0.5620940007642339,
      "skeleton_spread": 0.4904470768055025,
      "temporal_motion_energy": 0.4912113106610623
    },
    "oracle_move_episode_count": 5234,
    "future_candidate_skeleton_used": true,
    "oracle_perception_quality_upper_bound": true,
    "deployable": false,
    "source_artifact_granularity": "viewpoint-level mean sequence confidence; per-joint confidence is unavailable; valid_joint_fraction is derived from finite 3D pose values"
  },
  "method_c_future_recognition": {
    "status": "METHOD_C_NOT_AVAILABLE",
    "reason": "canonical skeleton archives contain no per-view recognition probabilities, entropy, or GT-class outputs"
  },
  "method_d_body_view_regression": {
    "model": {
      "architecture": "base Linear(569,128) + body_view Linear(23,64) -> concat(192) -> Linear(192,128)->GELU->Linear(128,64)->GELU->Linear(64,1)",
      "epochs": 20,
      "batch_size": 512,
      "learning_rate": 0.001,
      "loss": "SmoothL1Loss",
      "train_final_mae": 2.173166513442993
    },
    "candidate_level": {
      "n": 18648,
      "mae": 3.1163306267301354,
      "rmse": 4.715864050384306,
      "pearson": 0.3830532124236458,
      "spearman": 0.28717897449042606
    },
    "episode_level": {
      "three_way_accuracy": 0.45883802094025866,
      "binary_move_stay_accuracy": 0.5823239581194827,
      "both_move_candidate_hit": 0.5661738189686261,
      "selected_action_mean_true_utility": -0.011774667978382744,
      "harmful_move_count": 1942,
      "missed_beneficial_move_count": 2704
    },
    "high_margin": {
      "0.25": {
        "count": 5109,
        "three_way_accuracy": 0.5079271873165003,
        "binary_accuracy": 0.6218438050499119
      },
      "0.5": {
        "count": 4529,
        "three_way_accuracy": 0.522852726871274,
        "binary_accuracy": 0.6345771693530581
      },
      "1.0": {
        "count": 3821,
        "three_way_accuracy": 0.5404344412457471,
        "binary_accuracy": 0.6456425019628369
      },
      "2.0": {
        "count": 2826,
        "three_way_accuracy": 0.5651096956829441,
        "binary_accuracy": 0.6684359518754424
      }
    },
    "future_candidate_rgb_used": false,
    "future_candidate_depth_used": false,
    "future_candidate_semantic_used": false,
    "future_candidate_skeleton_used": false,
    "test_used": false
  },
  "comparison_references": {
    "exp030_a0_winner": 0.509744,
    "exp030_method_d_full_scene_winner": 0.559037,
    "exp030_method_b_candidate_hit": 0.582401
  },
  "leakage_flags": {
    "future_candidate_rgb_used": false,
    "future_candidate_depth_used": false,
    "future_candidate_semantic_used": false,
    "future_candidate_skeleton_used": true,
    "oracle_perception_quality_upper_bound": true,
    "deployable": false,
    "test_used": false
  },
  "provenance": {
    "source_commit": "933bb84d01006f642e5fc2405e66d0ed2771d285",
    "stage_d_train_features_sha256": "eb97b380da6ecd6a121144f8c29af523b6848eb7eaaf00573b7fa8b982516817",
    "stage_d_val_features_sha256": "4e78731acc35d356fcfa89e20c6cb119ec68da38b37342f45ef2d6336f36a72d",
    "runtime": "/home/zxf/WorkSpace/code/data/ActiveView/datasets/policy_v11_5/experiments/stage_d/EXP031_human_viewpoint_sensitivity"
  },
  "scientific_decision": {
    "case": "CASE_D",
    "basis": "descriptive comparison with frozen EXP030 candidate-hit reference; no Val tuning or arbitrary optimization threshold",
    "trajectory_rollout_performed": false
  }
}
