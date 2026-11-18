# Phase 3b trajectory inspection

All 20 real trajectories were inspected for loops, searched-location revisits, wrong object types, subgoal undoing, appliance repetition, and use of recent actions. Notes summarize observable behavior; no hidden reasoning or oracle plans were accessed. Environment success is the sole scoring signal.

Manifest: `alfworld-26a02eb5d85c76bb8171d23d438b793fabda50667afb9cdcdd220a94265481a5`. Rows follow its exact fixed order. Full task IDs and both old/new trajectory IDs are in `comparison.json`.

| # | Candidate trajectory ID | Outcome | Executed steps | Qualitative observation |
|---|---|---|---:|---|
| 1 | `e3df8df3-ba12-4867-b1df-92bcefb4ca0e` | success | 32 | Success after redundant cabinet/fridge visits. Correct apple selected. Repeated microwave open/close/examine actions before taking the apple back, heating it, and placing it in the garbagecan. |
| 2 | `a5215508-a306-4650-8f0b-0919077e4007` | success | 19 | Correct mug acquired; repeats microwave manipulation/examination before taking it back and using the heat command. Completes the placement without a sustained take/place loop. |
| 3 | `6331fb13-b5ea-41be-8491-d610b08a4fcf` | success | 6 | Turns on the desk lamp, finds/takes a CD, and returns to the lit desk. Environment confirms success on arrival; recent actions support the completed light/object sequence. |
| 4 | `9b41754f-b175-4e95-aa80-9d42075ff8a4` | environment_failure | 50 | Failure: cycles through cabinets, countertop, and two sinks for 50 steps. No target object acquired. Available locations such as the toilet and garbagecan remain unvisited; search coverage is not retained beyond the short window. |
| 5 | `25af2b47-96e7-4aae-8df7-89e826c3f204` | success | 18 | Finds two soapbars and deposits distinct objects in the garbagecan. Does not undo the first placement. Some redundant container search precedes a coherent two-object sequence. |
| 6 | `013e2e67-d677-45b0-ae78-e7dfc74b7cee` | success | 12 | Finds and places two distinct peppershakers in the drawer, retaining the recent placement subgoal and avoiding the original take/replace cycle. |
| 7 | `06aea203-b4a7-458b-87a3-dc9118dec8fb` | environment_failure | 50 | Failure: repeatedly revisits fridge/cabinets and a few drawers without finding the egg. Many available locations, including countertops, remain unvisited. The loop spans more than four pairs. |
| 8 | `f4c63db9-3ace-455c-98c8-5d862ddd6832` | success | 14 | After redundant initial search, takes the correct lettuce, cools it in the fridge, and returns it to the countertop. Tracks the recent acquisition and cooling subgoal. |
| 9 | `0f64ab9c-dc83-4920-bb3a-b75efb7d3ec2` | success | 25 | Succeeds, but briefly retakes/replaces the first deposited peppershaker after intervening search. Also relocates one correct-type instance before acquiring another. Older subgoal tracking is imperfect. |
| 10 | `d0bf07fa-65f2-40e5-9300-ade00e48d1cb` | environment_failure | 50 | Failure: repeatedly checks fridge/cabinets after limited exploration, never acquiring the tomato. This is incomplete search and revisitation, not an incorrect-object manipulation failure. |
| 11 | `8f8c91ca-2553-4001-899f-5562033ec5bd` | success | 18 | Redundant cabinet search eventually finds a mug; the take/clean/place sequence is coherent and succeeds. |
| 12 | `6d63161e-81a6-462c-8e8d-df7353f3c55f` | success | 18 | Some initial redundant cabinet search, then deposits two distinct soapbars in the cabinet without undoing the first deposit. |
| 13 | `fec20e56-4a41-4f4d-a95e-0783538bdc43` | success | 4 | Direct four-step soapbottle acquisition and toilet placement. No incorrect soapbar manipulation or placement-undo loop. |
| 14 | `53b739e4-88c8-4253-a1b8-3b36fcda8f7f` | success | 28 | Extensive redundant search before reaching the second countertop. Once the spatula is found, cleaning and placement complete coherently. |
| 15 | `fd18aa51-edc5-411b-bdb7-7bc0465d48dd` | environment_failure | 50 | Failure: repeatedly searches cabinets/drawers, reaching the first countertop only at the step limit. No target acquired; search coverage and priorities remain weak. |
| 16 | `e664074f-c4b2-47aa-8fce-0f1c793dbe9b` | success | 7 | Direct seven-step sequence acquires soapbar, cleans it, opens the cabinet, and places it. Recent state and subgoal tracking appear effective. |
| 17 | `65c938e1-6c71-4166-ab38-0e4268d84ec0` | success | 21 | Searches several containers before finding the vase on a dresser, then places it in the safe. No wrong watch manipulation or take/place loop. |
| 18 | `f1e71518-8796-493f-8c5d-7c05adb3b891` | success | 10 | Finds a mug, cools it, and places it in the cabinet. No wrong plate manipulation or non-admissible self-navigation. |
| 19 | `a19d0af9-3cb5-46c1-97b7-f328b76d4112` | success | 5 | Five-step peppershaker acquisition and drawer placement. No incorrect saltshaker manipulation or placement undo. |
| 20 | `8c427d95-639c-43fe-9cf3-ecaa6c4d925a` | success | 5 | Searches shelves, acquires a mug, and places it on the desk in five steps. Avoids the original two-desk loop. |

Recent acquisition and transformation state is generally tracked: successful runs complete distinct-object placements and take/clean-or-cool/place sequences. No incorrect object-type manipulation was observed. However, older search coverage is repeatedly lost or ignored, and row 9 briefly undoes an older placement. Rows 1–2 also show appliance-procedure confusion within recent context. The four failures (4, 7, 10, 15) are therefore not convincingly pure planning failures independent of missing working state: both search policy and the short context window remain plausible contributors.

Gate: **Suitable** for a subsequent controlled memory pilot. Sixteen successful trajectories cover all six task families; four failures retain headroom. This small diagnostic does not establish stable population performance. No Phase 4 work was performed.
