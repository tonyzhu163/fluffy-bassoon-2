Pinned RoboTwin dca9ec682688821c42944eacd7f5dd9bfb15396f, envs/_base_task.py SHA6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b.

Base_Task.compress_path444–451 keeps the first point and successive points differing by more than atol1e-5/rtol0. If only one point remains, it appends a copy plus Gaussian noise std2e-4. Base_Task.downsample_trajectory453–460 retains positions/velocities when≤500 points; otherwise it selects unique integer linspace indices from first to last. Exact source snippets are saved in the JSON companion.

Already-saved teacher00/01/02 reset JSONs and result.reset_info preserve seeds0/1/2, instructions, public observation/features hashes and render PCI, but contain no merged env.task_config/get_info. The nominated runtime output directory does not exist. No constructor/import/replay was used to recreate missing values. These source facts identify a convention difference requiring diagnosis; they do not establish the cause of phone02 failure.
