# EXP034 overnight audit

```json
{
  "experiment_id": "EXP034",
  "status": "COMPLETED",
  "split": "val",
  "test_used": false,
  "training_performed": false,
  "perception_regenerated": false,
  "habitat_rendering_performed": false,
  "stgcn_retrained": false,
  "models": {
    "model_0_current_legal_base": {
      "n": 18648,
      "mae": 3.2775596573608077,
      "rmse": 4.631524077413028,
      "pearson": 0.3509194683678189,
      "spearman": 0.2542191027379549,
      "r2": 0.12103307791226481
    },
    "model_1_true_pose_error": {
      "status": "BLOCKED"
    },
    "model_2_future_recognition_quality": {
      "n": 18648,
      "mae": 2.4662439906560287,
      "rmse": 3.2947286435949925,
      "pearson": 0.747714128462263,
      "spearman": 0.5888751884304113,
      "r2": 0.5552008152091323
    },
    "model_3_pose_plus_recognition": {
      "status": "BLOCKED"
    }
  },
  "incremental_delta_r2": {
    "model_2_minus_model_0": 0.43416773729686753
  },
  "conditional_utility_variance": {
    "action_class=11": {
      "n": 2108,
      "mean": -1.1800576981343491,
      "std": 5.085183053080145,
      "iqr": 3.9386641830205917,
      "move_fraction": 0.5617059891107078
    },
    "scene=00006-HkseAnWCgqk": {
      "n": 978,
      "mean": -0.7536688017297771,
      "std": 4.232633321192207,
      "iqr": 2.071897614747286,
      "move_fraction": 0.5551537070524413
    },
    "region=bedroom": {
      "n": 3465,
      "mean": -1.8556856480533908,
      "std": 5.159661028609026,
      "iqr": 3.949162721633911,
      "move_fraction": 0.5056421278882322
    },
    "current_correct=False": {
      "n": 13645,
      "mean": -1.554118301585267,
      "std": 5.256453862641188,
      "iqr": 4.198885470628738,
      "move_fraction": 0.5478297513695743
    },
    "current_confidence_quartile=2": {
      "n": 3580,
      "mean": -1.192464771633566,
      "std": 4.684084890652889,
      "iqr": 3.1818502992391586,
      "move_fraction": 0.5547835382148584
    },
    "current_correct=True": {
      "n": 5003,
      "mean": -0.6525838092855589,
      "std": 3.8746575928634,
      "iqr": 0.8834053426980972,
      "move_fraction": 0.6012199771254289
    },
    "action_class=9": {
      "n": 1554,
      "mean": -1.6262837767000997,
      "std": 5.575495569630229,
      "iqr": 3.380287751555443,
      "move_fraction": 0.5450061652281134
    },
    "current_confidence_quartile=0": {
      "n": 7184,
      "mean": -1.3276265705706431,
      "std": 4.88029123740152,
      "iqr": 3.024363074451685,
      "move_fraction": 0.5703249866808737
    },
    "current_confidence_quartile=3": {
      "n": 4386,
      "mean": -1.2908408595227434,
      "std": 5.100039001009113,
      "iqr": 3.7561392784118652,
      "move_fraction": 0.5681818181818182
    },
    "current_confidence_quartile=1": {
      "n": 3498,
      "mean": -1.4301028724105775,
      "std": 5.107895576004905,
      "iqr": 3.340123862028122,
      "move_fraction": 0.5456533624931657
    },
    "action_class=12": {
      "n": 2116,
      "mean": -1.2771563858870154,
      "std": 4.64891755180256,
      "iqr": 3.262282259762287,
      "move_fraction": 0.5380434782608695
    },
    "action_class=14": {
      "n": 362,
      "mean": 0.06616736697300771,
      "std": 2.7623045148971475,
      "iqr": 0.5051105916500092,
      "move_fraction": 0.6084656084656085
    },
    "action_class=8": {
      "n": 1847,
      "mean": -2.3298645443807704,
      "std": 5.615738606364037,
      "iqr": 5.5354701690375805,
      "move_fraction": 0.5041407867494824
    },
    "action_class=6": {
      "n": 1223,
      "mean": -1.2471758468345797,
      "std": 4.506524425475,
      "iqr": 4.080714046955109,
      "move_fraction": 0.5117739403453689
    },
    "action_class=15": {
      "n": 940,
      "mean": -1.3933943039770151,
      "std": 5.812750418908973,
      "iqr": 7.758935272693634,
      "move_fraction": 0.5447154471544715
    },
    "action_class=7": {
      "n": 1796,
      "mean": -0.5048429214225638,
      "std": 3.3851857690694125,
      "iqr": 0.5978336622938514,
      "move_fraction": 0.6159574468085106
    },
    "action_class=0": {
      "n": 1657,
      "mean": -0.9258694418544383,
      "std": 4.154976234424732,
      "iqr": 1.2706369012594223,
      "move_fraction": 0.6550925925925926
    },
    "action_class=5": {
      "n": 1181,
      "mean": -1.5177824423965196,
      "std": 5.193607064468923,
      "iqr": 1.1670218519866467,
      "move_fraction": 0.5609756097560976
    },
    "action_class=10": {
      "n": 1582,
      "mean": -0.9652083032819563,
      "std": 3.5862351760318982,
      "iqr": 1.574930096976459,
      "move_fraction": 0.5845410628019324
    },
    "action_class=13": {
      "n": 978,
      "mean": -1.9189064852588191,
      "std": 6.219758407117912,
      "iqr": 5.968289032578468,
      "move_fraction": 0.538160469667319
    },
    "action_class=3": {
      "n": 432,
      "mean": -1.0328100735873535,
      "std": 6.027401557683459,
      "iqr": 7.287214815616608,
      "move_fraction": 0.5859030837004405
    },
    "action_class=4": {
      "n": 350,
      "mean": -1.199646703476979,
      "std": 4.570870449115598,
      "iqr": 3.410330442711711,
      "move_fraction": 0.5519125683060109
    },
    "action_class=1": {
      "n": 196,
      "mean": -1.1994791898743702,
      "std": 5.542218346037356,
      "iqr": 6.02412337064743,
      "move_fraction": 0.6504854368932039
    },
    "action_class=2": {
      "n": 326,
      "mean": -2.763787259666071,
      "std": 6.814803231790619,
      "iqr": 8.184215754270554,
      "move_fraction": 0.4470588235294118
    },
    "region=living_room": {
      "n": 4769,
      "mean": -0.6752583021093357,
      "std": 4.3350343055388665,
      "iqr": 2.5397769808769226,
      "move_fraction": 0.5807587016034416
    },
    "region=kitchen": {
      "n": 5242,
      "mean": -1.6901991449965241,
      "std": 5.2296404403677546,
      "iqr": 3.969962850213051,
      "move_fraction": 0.5493060628195763
    },
    "region=dining_area": {
      "n": 5172,
      "mean": -1.1524631654389828,
      "std": 4.936375284396001,
      "iqr": 2.890999734401703,
      "move_fraction": 0.5982211910286156
    },
    "scene=00062-ACZZiU6BXLz": {
      "n": 874,
      "mean": -1.8754189238898467,
      "std": 4.936793593118557,
      "iqr": 4.276475252583623,
      "move_fraction": 0.4574468085106383
    },
    "scene=00087-YY8rqV6L6rf": {
      "n": 898,
      "mean": -3.022546943625207,
      "std": 6.016194770243167,
      "iqr": 7.518366388278082,
      "move_fraction": 0.44543429844098
    },
    "scene=00096-6HRFAUDqpTb": {
      "n": 550,
      "mean": -0.7131342953255291,
      "std": 4.486058718793841,
      "iqr": 3.712996229529381,
      "move_fraction": 0.5927272727272728
    },
    "scene=00164-XfUxBGTFQQb": {
      "n": 748,
      "mean": -2.1164281375838545,
      "std": 5.735706749240572,
      "iqr": 4.628393538296223,
      "move_fraction": 0.43204868154158216
    },
    "scene=00172-bB6nKqfsb1z": {
      "n": 744,
      "mean": -0.19267516114661282,
      "std": 3.904028926632918,
      "iqr": 1.3355368450284004,
      "move_fraction": 0.6881720430107527
    },
    "scene=00250-U3oQjwTuMX8": {
      "n": 1086,
      "mean": -0.2810169424159539,
      "std": 3.99259819926257,
      "iqr": 1.7895507365465164,
      "move_fraction": 0.6685082872928176
    },
    "scene=00251-wsAYBFtQaL7": {
      "n": 1218,
      "mean": -1.1929836679661165,
      "std": 4.69397309978828,
      "iqr": 2.950063120573759,
      "move_fraction": 0.5796387520525451
    },
    "scene=00299-bdp1XNEdvmW": {
      "n": 623,
      "mean": -3.5341819032192947,
      "std": 6.862453946650029,
      "iqr": 8.46924439445138,
      "move_fraction": 0.44352617079889806
    },
    "scene=00326-u9rPN5cHWBg": {
      "n": 538,
      "mean": -2.1572916468423213,
      "std": 5.60038009626444,
      "iqr": 5.229073025286198,
      "move_fraction": 0.5650557620817844
    },
    "scene=00327-xgLmjqzoAzF": {
      "n": 1236,
      "mean": -2.5655971452940265,
      "std": 6.228342712822549,
      "iqr": 5.603912312537432,
      "move_fraction": 0.511326860841424
    },
    "scene=00417-nGhNxKrgBPb": {
      "n": 846,
      "mean": -0.7349473770021087,
      "std": 4.1967764339247395,
      "iqr": 3.0921694189310074,
      "move_fraction": 0.5910165484633569
    },
    "scene=00422-8wJuSPJ9FXG": {
      "n": 1068,
      "mean": -1.0783370600352666,
      "std": 4.3476080841837526,
      "iqr": 2.825722023844719,
      "move_fraction": 0.5543071161048689
    },
    "scene=00444-sX9xad6ULKc": {
      "n": 756,
      "mean": -0.8762486815250558,
      "std": 4.063332755292145,
      "iqr": 2.2786350399255753,
      "move_fraction": 0.6031746031746031
    },
    "scene=00475-g7hUFVNac26": {
      "n": 958,
      "mean": -1.2785674856316596,
      "std": 5.028942019254802,
      "iqr": 3.695967398583889,
      "move_fraction": 0.5845511482254697
    },
    "scene=00476-NtnvZSMK3en": {
      "n": 621,
      "mean": -0.9930098232562254,
      "std": 5.486916850128278,
      "iqr": 2.9407036304473877,
      "move_fraction": 0.5149863760217984
    },
    "scene=00487-erXNfWVjqZ8": {
      "n": 1150,
      "mean": -2.377318199715569,
      "std": 5.587703209768408,
      "iqr": 5.476448552682996,
      "move_fraction": 0.5113043478260869
    },
    "scene=00534-DBBESbk4Y3k": {
      "n": 1100,
      "mean": -0.48173931478434734,
      "std": 3.584124527832454,
      "iqr": 1.3709096536040306,
      "move_fraction": 0.6654545454545454
    },
    "scene=00592-CthA7sQNTPK": {
      "n": 474,
      "mean": -0.4685223102163344,
      "std": 4.42681995002548,
      "iqr": 2.688319742679596,
      "move_fraction": 0.5864978902953587
    },
    "scene=00643-ggNAcMh8JPT": {
      "n": 1052,
      "mean": -0.337338300119626,
      "std": 3.9217976872822957,
      "iqr": 1.581924706697464,
      "move_fraction": 0.6444866920152091
    },
    "scene=00750-E1NrAhMoqvB": {
      "n": 1130,
      "mean": -0.7537907451261227,
      "std": 3.850507730130413,
      "iqr": 2.349345773458481,
      "move_fraction": 0.6265486725663717
    }
  },
  "same_context_cross_motion_instability": {
    "definition": "rounded current legal base context; duplicate groups only",
    "duplicate_group_count": 10,
    "three_way_switch_rate": 0.0,
    "binary_switch_rate": 0.0
  },
  "leakage_flags": {
    "future_candidate_skeleton_used": false,
    "future_recognition_quality_used_as_model_input": false,
    "test_used": false
  },
  "provenance": {
    "source_commit": "d31a95ddef5edff0f3ce2314443314c06568dce1",
    "stage_d_feature_summary_sha256": "5ada9ca595aca421b77c507ee6ea3ccd19eda0008fda943f4483e5a41837adfb",
    "stage_b_train_utility_sha256": "c6f78582bcbeae5780ed3fde34f6a8a279ef73f495a57f2bea79f85b68b545c4",
    "stage_b_val_utility_sha256": "34f1d55724940a150ce55cf3babefdc2fcb700f6074093948916552958527729"
  }
}
```
