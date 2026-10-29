# Focused Linux qualification

Source `352edc407738ce14b6a0adb9100732326ca40902`; Baige job `job-gjkse1sb5gyj`.

Command: `CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_maniskill_noise.py tests/test_maniskill_pi05.py tests/test_maniskill_rotation_poc.py`.

Result: **31 passed in 8.82 seconds**. See `evidence/qualification.log`. AST and Bash syntax passed on macOS. The allocated GPU probe also passed; real simulator behavior is a separate ongoing qualification.
