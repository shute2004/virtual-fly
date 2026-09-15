# Flyppy population state diagnostic

- generated_at_utc: 2026-09-15T03:14:49+00:00
- experiment: `/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3`
- diagnostic mutates training artifacts: false

## Files

| artifact | exists | bytes | mtime UTC | path |
|---|---|---:|---|---|
| summary | yes | 16146 | 2026-09-15T00:11:12+00:00 | `artifacts/experiments/flyppy-v3/summary.json` |
| curriculum | yes | 421 | 2026-09-15T00:11:12+00:00 | `artifacts/experiments/flyppy-v3/curriculum-state.json` |
| population_state | no | - | - | `/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3/population-state.json` |
| commit_log | no | - | - | `/Users/<local-user>/Desktop/virtual-fly/artifacts/experiments/flyppy-v3/commit-log.jsonl` |
| trajectory | yes | 1096473 | 2026-09-15T00:09:48+00:00 | `artifacts/experiments/flyppy-v3/trajectory.jsonl` |
| checkpoint_manifest | yes | 402 | 2026-09-15T00:11:12+00:00 | `artifacts/experiments/flyppy-v3/checkpoint/manifest.json` |

## Summary

```json
{
  "backend": "gpu:Apple M1 (Metal)",
  "checkpoint_neural_step": 5757,
  "elapsed_seconds": 251.46552487509325,
  "episode_end": 71,
  "episode_start": 48,
  "episodes_this_run": 24,
  "experiment": "flyppy_v3_source_equivalent_whole_body_curriculum",
  "schema_version": 10
}
```

## Curriculum state

```json
{
  "consecutive_failures": 0,
  "curriculum_complete": false,
  "curriculum_episodes": 72,
  "curriculum_mode": "adaptive",
  "initial_speed_mm_s": 362.5,
  "schema_version": 4,
  "spawn_x_mm": 1.336500000000001,
  "spawn_z_mm": 9.98233224757348,
  "successful_first_gates": 41
}
```

## Population state

```json
{
  "missing": true
}
```

## Checkpoint manifest

```json
{
  "dataset": "male-cns:v1.0",
  "edge_count": 25582938,
  "neuron_count": 166700,
  "schema_version": 3,
  "step": 5757
}
```

## Commit log

- non-empty records: -
- latest episode: -
- latest commit weight version: -

```json
[]
```

## Trajectory

- non-empty records: 671
- latest episode: 71

```json
[
  {
    "aversive_stimulated": false,
    "boundary_ease_level": null,
    "collision": false,
    "collision_reason": null,
    "control_step": 70,
    "curriculum_mode": "adaptive",
    "environment_version": "v3",
    "episode": 71,
    "finished": false,
    "motor_boundary": "whole-body",
    "motor_periphery": {
      "active_motor_units": 60,
      "active_somatic_units": 110,
      "dlm_left": 0.9839528860371323,
      "dlm_right": 0.9886638310968027,
      "dvm_left": 0.9965914077811112,
      "dvm_right": 0.9965556715941969,
      "haltere_power": {
        "left": 0.617409681521248,
        "right": 0.5695364960518431
      },
      "leg_drive": {
        "lf:coxa_femur_pitch": 0.014476244998536725,
        "lf:femur_tibia_pitch": -0.0174283356065964,
        "lh:femur_tibia_pitch": -0.00023256585210251757,
        "lh:thorax_coxa_yaw": 0.0012665631670720767,
        "lm:coxa_femur_pitch": -0.003837817361332263,
        "lm:femur_tibia_pitch": -0.005737741198806767,
        "lm:thorax_coxa_yaw": -0.042315958135200726,
        "rf:coxa_femur_pitch": 0.8952432884679035,
        "rf:femur_tibia_pitch": 0.12653625770209376,
        "rf:thorax_coxa_yaw": 0.000972144694459165,
        "rh:coxa_femur_pitch": -0.04137708275218055,
        "rh:femur_tibia_pitch": -0.07744485859531347,
        "rh:thorax_coxa_yaw": -0.0014303939676170163,
        "rm:coxa_femur_pitch": -0.1420259253654762,
        "rm:femur_tibia_pitch": -0.0010949588320073866,
        "rm:thorax_coxa_yaw": -0.009159594744674315
      },
      "somatic_spikes": 18,
      "spikes": 3,
      "steering": {
        "left:b1": 0.7673020234105383,
        "left:b2": 0.814649091173963,
        "left:b3": 0.594565232389445,
        "left:hg1": 0.7929298659932748,
        "left:hg2": 0.8117361595143551,
        "left:hg3": 0.7690051620277238,
        "left:hg4": 0.8495828007658909,
        "left:i1": 0.9321947310607862,
        "left:i2": 0.9626408080164255,
        "left:iii1": 0.7952112473119326,
        "left:iii3": 0.9233394671198774,
        "right:b1": 0.8185203632069181,
        "right:b2": 0.7805521312179979,
        "right:b3": 0.8485365330727198,
        "right:hg1": 0.8247564801116681,
        "right:hg2": 0.8142350686704306,
        "right:hg3": 0.8469974586454491,
        "right:hg4": 0.9009480767828695,
        "right:i1": 0.601719454539817,
        "right:i2": 0.9271231596461565,
        "right:iii1": 0.7433512765048533,
        "right:iii3": 0.8227181869580183
      },
      "whole_body": true
    },
    "next_gate": 1,
    "passed_gate": false,
    "retinal_input": {
      "active_columns": 800,
      "active_photoreceptors": 3273,
      "max_current": 2.0,
      "mean_current": 0.8322290412266994
    },
    "reward_stimulated": false,
    "vx_mm_s": 406.6227063951217,
    "vy_mm_s": 14.698827196327116,
    "vz_mm_s": -95.61027287976084,
    "x_mm": 15.406671648418772,
    "y_mm": 0.10540177788676591,
    "z_mm": 8.917259031361992
  },
  {
    "aversive_stimulated": false,
    "boundary_ease_level": null,
    "collision": false,
    "collision_reason": null,
    "control_step": 80,
    "curriculum_mode": "adaptive",
    "environment_version": "v3",
    "episode": 71,
    "finished": false,
    "motor_boundary": "whole-body",
    "motor_periphery": {
      "active_motor_units": 60,
      "active_somatic_units": 113,
      "dlm_left": 0.9948116126273199,
      "dlm_right": 0.9962588875011874,
      "dvm_left": 0.9982929081126328,
      "dvm_right": 0.9982707162127377,
      "haltere_power": {
        "left": 0.6920994100319409,
        "right": 0.6559997908053112
      },
      "leg_drive": {
        "lf:coxa_femur_pitch": -0.030950036532357728,
        "lf:femur_tibia_pitch": -0.0032242815847672857,
        "lf:thorax_coxa_yaw": 0.0007400973931656063,
        "lh:femur_tibia_pitch": -0.05376799224720141,
        "lh:thorax_coxa_yaw": -0.0046894311362899455,
        "lm:coxa_femur_pitch": -0.01736602944519494,
        "lm:femur_tibia_pitch": -0.001088289913298901,
        "lm:thorax_coxa_yaw": -0.0013688865445746767,
        "rf:coxa_femur_pitch": 0.7885739962724255,
        "rf:femur_tibia_pitch": -0.022593814797314815,
        "rf:thorax_coxa_yaw": 0.04408414046953002,
        "rh:coxa_femur_pitch": -0.015207797320984984,
        "rh:femur_tibia_pitch": -0.014982590335939738,
        "rh:thorax_coxa_yaw": -0.010776844864332724,
        "rm:coxa_femur_pitch": -0.2399091461203422,
        "rm:femur_tibia_pitch": -0.01241999843905539,
        "rm:thorax_coxa_yaw": -0.011248423806862973
      },
      "somatic_spikes": 11,
      "spikes": 10,
      "steering": {
        "left:b1": 0.8847768195960957,
        "left:b2": 0.885479114588882,
        "left:b3": 0.8089166009712596,
        "left:hg1": 0.9570061919422287,
        "left:hg2": 0.7997788847602315,
        "left:hg3": 0.8580816695813073,
        "left:hg4": 0.8660496775825792,
        "left:i1": 0.8246943726995961,
        "left:i2": 0.8147012560970338,
        "left:iii1": 0.8643331695232813,
        "left:iii3": 0.6087028921930496,
        "right:b1": 0.9188098229425551,
        "right:b2": 0.884973357308073,
        "right:b3": 0.6681043507939702,
        "right:hg1": 0.716521822520475,
        "right:hg2": 0.8000259921272612,
        "right:hg3": 0.8938982928908799,
        "right:hg4": 0.8867591800361977,
        "right:i1": 0.7790111530192263,
        "right:i2": 0.8993115748299982,
        "right:iii1": 0.7930165707545265,
        "right:iii3": 0.8855988027941532
      },
      "whole_body": true
    },
    "next_gate": 1,
    "passed_gate": false,
    "retinal_input": {
      "active_columns": 803,
      "active_photoreceptors": 3276,
      "max_current": 2.0,
      "mean_current": 0.9345859705409016
    },
    "reward_stimulated": false,
    "vx_mm_s": 429.9729224617037,
    "vy_mm_s": 29.634442052632323,
    "vz_mm_s": -99.7303566944925,
    "x_mm": 17.492065030647645,
    "y_mm": 0.19506990643879707,
    "z_mm": 8.354709377963765
  },
  {
    "aversive_stimulated": false,
    "boundary_ease_level": null,
    "collision": false,
    "collision_reason": null,
    "control_step": 90,
    "curriculum_mode": "adaptive",
    "environment_version": "v3",
    "episode": 71,
    "finished": false,
    "motor_boundary": "whole-body",
    "motor_periphery": {
      "active_motor_units": 60,
      "active_somatic_units": 116,
      "dlm_left": 0.9990441346121969,
      "dlm_right": 0.9992947540232058,
      "dvm_left": 0.9996825054656703,
      "dvm_right": 0.9996778287683392,
      "haltere_power": {
        "left": 0.6783949235096967,
        "right": 0.6844175152238817
      },
      "leg_drive": {
        "lf:coxa_femur_pitch": -0.11050932726091356,
        "lf:femur_tibia_pitch": -0.0011486798108326646,
        "lf:thorax_coxa_yaw": 0.0001984658384204252,
        "lh:femur_tibia_pitch": -0.011422184172806826,
        "lh:thorax_coxa_yaw": -0.0035981874730763863,
        "lm:coxa_femur_pitch": -0.0012148602288702959,
        "lm:femur_tibia_pitch": -0.04563128446876874,
        "lm:thorax_coxa_yaw": -0.0066085602711131175,
        "rf:coxa_femur_pitch": 0.8743878064846247,
        "rf:femur_tibia_pitch": -0.02501305568982448,
        "rf:thorax_coxa_yaw": 0.03550062727123493,
        "rh:coxa_femur_pitch": -0.016324693571615367,
        "rh:femur_tibia_pitch": -0.0091414829103873,
        "rh:thorax_coxa_yaw": -0.0037994206537075303,
        "rm:coxa_femur_pitch": -0.08621894450142908,
        "rm:femur_tibia_pitch": -0.005302366505575251,
        "rm:thorax_coxa_yaw": -0.008318173614282554
      },
      "somatic_spikes": 14,
      "spikes": 17,
      "steering": {
        "left:b1": 0.9624209184746928,
        "left:b2": 0.9624313355560494,
        "left:b3": 0.5332706898038939,
        "left:hg1": 0.9594399220039189,
        "left:hg2": 0.5272467360103429,
        "left:hg3": 0.774998239320774,
        "left:hg4": 0.9663679246769806,
        "left:i1": 0.8856281153876608,
        "left:i2": 0.8854798883467092,
        "left:iii1": 0.569803543379675,
        "left:iii3": 0.8198960460158989,
        "right:b1": 0.9408575150066512,
        "right:b2": 0.7776574532953962,
        "right:b3": 0.7280468955992553,
        "right:hg1": 0.8840236012342751,
        "right:hg2": 0.5274096392267109,
        "right:hg3": 0.7947627529898587,
        "right:hg4": 0.6967397660901845,
        "right:i1": 0.8224222138327953,
        "right:i2": 0.8709671618355469,
        "right:iii1": 0.8851582412700141,
        "right:iii3": 0.583822712858776
      },
      "whole_body": true
    },
    "next_gate": 1,
    "passed_gate": false,
    "retinal_input": {
      "active_columns": 805,
      "active_photoreceptors": 3282,
      "max_current": 2.0,
      "mean_current": 0.9244199276210041
    },
    "reward_stimulated": false,
    "vx_mm_s": 454.8991125184109,
    "vy_mm_s": 41.40710545416931,
    "vz_mm_s": -139.45459492387195,
    "x_mm": 19.6446479314863,
    "y_mm": 0.3214423883422856,
    "z_mm": 7.582285403233678
  },
  {
    "aversive_stimulated": false,
    "boundary_ease_level": null,
    "collision": false,
    "collision_reason": null,
    "control_step": 100,
    "curriculum_mode": "adaptive",
    "environment_version": "v3",
    "episode": 71,
    "finished": false,
    "motor_boundary": "whole-body",
    "motor_periphery": {
      "active_motor_units": 60,
      "active_somatic_units": 118,
      "dlm_left": 0.9997104345968786,
      "dlm_right": 0.999808119374411,
      "dvm_left": 0.999822008855067,
      "dvm_right": 0.9998778302656078,
      "haltere_power": {
        "left": 0.6649618040022427,
        "right": 0.7091673036110125
      },
      "leg_drive": {
        "lf:coxa_femur_pitch": -0.06265964693019044,
        "lf:femur_tibia_pitch": 0.0023621416390756655,
        "lf:thorax_coxa_yaw": 0.0019966962183800563,
        "lh:coxa_femur_pitch": 0.6682161691290174,
        "lh:femur_tibia_pitch": -0.03559395396024312,
        "lh:thorax_coxa_yaw": -0.005661508354828926,
        "lm:coxa_femur_pitch": -0.0006068520624421536,
        "lm:femur_tibia_pitch": -0.0053475153821935795,
        "lm:thorax_coxa_yaw": -0.032604593194145326,
        "rf:coxa_femur_pitch": 0.9967373230520767,
        "rf:femur_tibia_pitch": -0.00791959870694181,
        "rf:thorax_coxa_yaw": 0.002101272269051435,
        "rh:coxa_femur_pitch": -0.05242507109883643,
        "rh:femur_tibia_pitch": -0.04941506706233756,
        "rh:thorax_coxa_yaw": -0.005039083699925451,
        "rm:coxa_femur_pitch": -0.19951534611586785,
        "rm:femur_tibia_pitch": -0.037249190722188574,
        "rm:thorax_coxa_yaw": -0.017319454551732782
      },
      "somatic_spikes": 14,
      "spikes": 20,
      "steering": {
        "left:b1": 0.8146795120790238,
        "left:b2": 0.8146805421835157,
        "left:b3": 0.8187771666314893,
        "left:hg1": 0.8955661423000086,
        "left:hg2": 0.7422832648058273,
        "left:hg3": 0.8970551189422471,
        "left:hg4": 0.9375715109588855,
        "left:i1": 0.7777221997647487,
        "left:i2": 0.9624313470331329,
        "left:iii1": 0.3756376470280627,
        "left:iii3": 0.9574061789762536,
        "right:b1": 0.8750754776031959,
        "right:b2": 0.8224021345209669,
        "right:b3": 0.47995809427032965,
        "right:hg1": 0.7493983070091563,
        "right:hg2": 0.7422993736686805,
        "right:hg3": 0.8973482842773004,
        "right:hg4": 0.75904376860049,
        "right:i1": 0.8008355869377184,
        "right:i2": 0.9622160810062822,
        "right:iii1": 0.7495105071309881,
        "right:iii3": 0.6926968232364202
      },
      "whole_body": true
    },
    "next_gate": 1,
    "passed_gate": false,
    "retinal_input": {
      "active_columns": 821,
      "active_photoreceptors": 3326,
      "max_current": 2.0,
      "mean_current": 0.7927751656846143
    },
    "reward_stimulated": false,
    "vx_mm_s": 461.6328873723757,
    "vy_mm_s": 44.102332014656504,
    "vz_mm_s": -212.6487313589957,
    "x_mm": 21.839347957028686,
    "y_mm": 0.4722925287759123,
    "z_mm": 6.543133932253959
  },
  {
    "aversive_stimulated": true,
    "boundary_ease_level": null,
    "collision": true,
    "collision_reason": "gate",
    "control_step": 103,
    "curriculum_mode": "adaptive",
    "environment_version": "v3",
    "episode": 71,
    "finished": false,
    "motor_boundary": "whole-body",
    "motor_periphery": {
      "active_motor_units": 60,
      "active_somatic_units": 119,
      "dlm_left": 0.9997756867426663,
      "dlm_right": 0.9997804006815303,
      "dvm_left": 0.9998949956213351,
      "dvm_right": 0.9998797542954982,
      "haltere_power": {
        "left": 0.7016659011573882,
        "right": 0.739854993313232
      },
      "leg_drive": {
        "lf:coxa_femur_pitch": -0.09114351788928432,
        "lf:femur_tibia_pitch": 0.0002865001444909421,
        "lf:thorax_coxa_yaw": 0.005774162861063581,
        "lh:coxa_femur_pitch": 0.6293022888532425,
        "lh:femur_tibia_pitch": -0.051569125609420285,
        "lh:thorax_coxa_yaw": -0.002973961885577303,
        "lm:coxa_femur_pitch": -0.0025367168563447118,
        "lm:femur_tibia_pitch": -0.004817596106926003,
        "lm:thorax_coxa_yaw": -0.016678150719066864,
        "rf:coxa_femur_pitch": 0.9820817346177367,
        "rf:femur_tibia_pitch": -0.017541274373571003,
        "rf:thorax_coxa_yaw": 0.003751918709766766,
        "rh:coxa_femur_pitch": -0.08057775095305453,
        "rh:femur_tibia_pitch": -0.07954361341525995,
        "rh:thorax_coxa_yaw": 0.000696041533298053,
        "rm:coxa_femur_pitch": -0.23316558615428107,
        "rm:femur_tibia_pitch": -0.05820186173692754,
        "rm:thorax_coxa_yaw": -0.03413900606364961
      },
      "somatic_spikes": 10,
      "spikes": 4,
      "steering": {
        "left:b1": 0.9578428219013302,
        "left:b2": 0.9231539968037011,
        "left:b3": 0.8904229994538168,
        "left:hg1": 0.7903343466393925,
        "left:hg2": 0.913570440847496,
        "left:hg3": 0.9340582931298957,
        "left:hg4": 0.9394216316986838,
        "left:i1": 0.884988367289475,
        "left:i2": 0.8493426827070597,
        "left:iii1": 0.865035897542232,
        "left:iii3": 0.8449079874618964,
        "right:b1": 0.9311487483196508,
        "right:b2": 0.9241761390018244,
        "right:b3": 0.8455719821695205,
        "right:hg1": 0.6613416847377201,
        "right:hg2": 0.9135725732507317,
        "right:hg3": 0.9008238146470725,
        "right:hg4": 0.915789104750178,
        "right:i1": 0.7067349249520529,
        "right:i2": 0.8491527111051322,
        "right:iii1": 0.8812538575845806,
        "right:iii3": 0.6113028009363298
      },
      "whole_body": true
    },
    "next_gate": 1,
    "passed_gate": false,
    "retinal_input": {
      "active_columns": 823,
      "active_photoreceptors": 3331,
      "max_current": 2.0,
      "mean_current": 0.7676799421652208
    },
    "reward_stimulated": false,
    "vx_mm_s": 395.8007212832745,
    "vy_mm_s": -39.18941844889945,
    "vz_mm_s": -295.1832066758403,
    "x_mm": 22.498636949597792,
    "y_mm": 0.5221331263237375,
    "z_mm": 6.173449668737803
  }
]
```

## Consistency findings

- summary.json is not a shared-weight population summary
