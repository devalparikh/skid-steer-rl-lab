# Executed results

Two fixed-budget CPU PPO runs. Test seeds 1000–1019; deterministic policy evaluation.

| Training seed | Checkpoint | Successes / 20 | Mean return | Mean final distance (m) |
| --- | --- | --- | --- | --- |
| 7 | initial | 0 | -2.40 | 3.26 |
| 7 | final | 1 | 5.60 | 1.74 |
| 19 | initial | 0 | -2.40 | 3.26 |
| 19 | final | 0 | 4.62 | 1.78 |

Both runs reduce final target distance, but parking remains mostly unsolved. Seed 7 succeeds in 1/20 episodes and seed 19 in 0/20. Reward improvement is not task completion. The results do not establish reliable control or transfer to hardware.

Videos use the predetermined first test seed (1000), so the seed-7 video happens to show its only successful test episode. Read the aggregate results above before interpreting that clip.

Training times (including intermediate validation): seed 7: 77.25 s, seed 19: 78.87 s. These are observations on this shared 4-vCPU/8-GB VM, not performance benchmarks.

No environment or reward changes were made between these two runs. The small TypeScript balance toy in the article is a separate experiment.
