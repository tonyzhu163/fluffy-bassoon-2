# Why Sidney missed the clean phone-stand baseline

The original Sidney SFT checkpoint succeeded on 23 of 100 clean scenes in our SAPIEN 3.0.1 deployment. Moving the near clipping plane on both wrist cameras from 10 cm to 1 cm raised success to **75/100** on the same scenes. The model card reports **66/100** on easy phone stand. The uploader does not publish the exact episode list or camera settings. [Published task results](https://huggingface.co/SidneyXie/pi05_robotwin#evaluation-results).

| Outcome on the same clean scene | Scenes |
| --- | ---: |
| Success with either camera setting | 20 |
| Success only at 1 cm | 55 |
| Success only at 10 cm | 3 |
| Failure with either setting | 22 |

The 52 percentage-point gain has an exact two-sided paired test p-value of 2.26 × 10⁻¹³. Scene IDs, actor assets, prompts, reset joint states, head-camera pixels, controller, policy weights, sampling noise and 400-target limit matched. The first action chunk changed on every scene. The independent readback verified all 100 resets, all 425 corrected action chunks, and the renderer's device assignment. [Clean readback](WRIST-CLEAN-READBACK.json).

Training demonstrations show the gripper fingertips in both wrist views. At the original clipping distance the simulator excluded most fingertip geometry. At 1 cm the fingertips appear again. For the fingertip region chosen in the image, the median number of dark pixels in left and right views was 1,074 and 1,092.5 across 16 training resets, 19 and 17 in seven original simulator resets, and 1,036 and 1,062 in the corresponding corrected resets. The intervention changes everything visible within the recovered depth range. It supports an effect of camera clipping, without isolating individual pixels used by the network. [Training and paired simulation images](wrist-comparison.jpg), [image measurements](FINGERTIP-REGION.json), [RLinf camera documentation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html#installation).

The reference LeRobot loader matched all 813 saved checkpoint tensors after conversion to its parameter dtype. The observation adapter reproduced the reference preprocessor's images, state, language tokens and attention masks exactly on 16 actual resets. Both action tests used the same frozen model and original mean/std processors. [Loader audit](LOADER.json), [preprocessor audit](PROCESSOR-PARITY.json).

## Randomized scenes

On the 100 randomized scenes, the same wrist-only correction raised success from **31/100 to 87/100**. Twenty-nine scenes succeeded with both settings, 58 only with the correction, two only with original clipping and 11 with neither. Exact paired p = 3.18 × 10⁻¹⁵. All 100 head-camera images and float32 reset joint vectors matched exactly, and all 100 first action chunks changed. The verifier checked 365 corrected action chunks. The Sidney model card reports only the clean result for this task. The initial randomized job exited before model loading after a renderer import moved. The corrected job completed all cases. [Randomized readback](WRIST-RANDOM-READBACK.json), [startup failure](WRIST-RANDOM-FIRST-EXIT.json).
