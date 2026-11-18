# Diagnostic trajectory inspection

Real ALFWorld/OpenAI pilot, 2026-10-25. All 20 trajectories were inspected for task goals, action patterns, observations, and parse failures. These qualitative notes do not alter deterministic environment scoring.

Manifest: `alfworld-26a02eb5d85c76bb8171d23d438b793fabda50667afb9cdcdd220a94265481a5`. Source trajectories are in `trajectories.jsonl`; row order follows the fixed manifest.

| # | Source trajectory ID | Outcome | Executed steps | Observed pattern |
|---|---|---|---:|---|
| 1 | `1183ddb8-51cd-45f5-96d8-f07fb2438b0d` | environment_failure | 50 | Apple goal: takes a bowl; cycles between fridge/microwave and repeatedly places/takes the bowl. |
| 2 | `8a0db245-b23e-4810-91ea-cda0e25d5d1c` | aborted | 43 | Heated-mug goal: repeatedly heats a cup and navigates microwave/cabinet; quoted action aborts. |
| 3 | `49c995d3-c0fa-4106-bb2a-ad9942ac357d` | environment_failure | 50 | CD/light goal: uses desklamp 48 times without completing the object-lighting objective. |
| 4 | `c69b6a9b-2b4e-4c6b-a1aa-606891982982` | environment_failure | 50 | Clean-soapbar goal: alternates between two sink basins for all 50 actions. |
| 5 | `e42734ee-9399-4a62-b87b-290914f2ab2b` | environment_failure | 50 | Two-soapbar goal: takes a soapbottle and repeatedly places/takes it in the garbagecan. |
| 6 | `95ce3a08-ffe5-4822-bb7d-bdae63526454` | environment_failure | 50 | Two-peppershaker goal: repeatedly takes/places one peppershaker in its cabinet rather than the drawer. |
| 7 | `98c312df-dfd5-4488-955d-069be879a12c` | environment_failure | 50 | Heated-egg goal: opens appliances, then cycles between microwave and fridge. |
| 8 | `20e5c3f7-c670-4108-ae5b-2599ec73474e` | aborted | 3 | Cool-lettuce goal: takes and immediately replaces lettuce; non-admissible put command aborts. |
| 9 | `7c066d51-00ae-484e-9b81-00e3b7eb4f3d` | environment_failure | 50 | Two-peppershaker goal: repeatedly takes/places a cellphone and visits drawers. |
| 10 | `25124717-0474-4c0e-8390-27c1ae893330` | environment_failure | 50 | Cool-tomato goal: takes lettuce, then repeatedly navigates between microwave and fridge. |
| 11 | `3c8fc977-f1ef-46fe-b6d3-f34b73cb3b7f` | aborted | 24 | Clean-mug goal: cycles between coffeemachine/cabinet; later attempts non-admissible navigation to its current countertop. |
| 12 | `a21335d8-85c1-443d-9a39-a220f2047eab` | aborted | 9 | Two-soapbar goal: places and retakes the same soapbar; quoted navigation command aborts. |
| 13 | `eced25c4-455c-4bbd-a1cc-efb936dda3c8` | environment_failure | 50 | Soapbottle goal: repeatedly takes/places a soapbar in the toilet. |
| 14 | `a992b052-0484-4ef5-9fe4-73b904a4f453` | environment_failure | 50 | Clean-spatula goal: takes a saltshaker and repeatedly places/takes it in a drawer. |
| 15 | `3d61be17-cbc7-4d51-8f77-f2c1711c5f5e` | aborted | 4 | Peppershaker goal: takes/places a dishsponge; non-admissible navigation to its current drawer aborts. |
| 16 | `1c36d8cd-3e49-4670-b94e-f1f478b19d68` | environment_failure | 50 | Clean-soapbar goal: finds soapbar but places/retakes it in cabinet without a cleaning sequence. |
| 17 | `b18c0535-f34e-4a7b-b8b2-2709f7a71922` | environment_failure | 50 | Vase goal: repeatedly takes/places a watch in the safe. |
| 18 | `65eac29f-c89c-46f8-9a80-7c10c375c456` | aborted | 5 | Cool-mug goal: takes/places a plate; non-admissible navigation to its current cabinet aborts. |
| 19 | `2c9cac5a-9fa5-412c-92d5-112523d229a5` | environment_failure | 50 | Peppershaker goal: repeatedly takes/places a saltshaker in the drawer. |
| 20 | `2273cca6-2955-4429-8f27-e1598d81cf65` | environment_failure | 50 | Mug goal: alternates between two desks for all 50 actions. |

Recurring structures include navigation, opening containers, taking/placing objects, and appliance actions. They could motivate future procedural assistance, but here the repeated patterns are mostly unsuccessful loops; this pilot yields no successful trajectories for a successful-episode memory baseline.

The configured model/prompt setup is unsuitable as the current comparison baseline: 0/14 completed tasks succeeded and 6/20 attempted tasks aborted. This pilot cannot isolate model capability from the deliberately frozen, stateless prompt. No prompt/model changes were made, and Phase 4 was not started.
