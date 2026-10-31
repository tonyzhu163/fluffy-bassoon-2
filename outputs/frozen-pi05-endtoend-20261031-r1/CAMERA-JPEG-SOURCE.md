# Camera and JPEG source — pinned static handoff

RoboTwin pin `dca9ec682688821c42944eacd7f5dd9bfb15396f`:

- `envs/camera/camera.py`, SHA `49ff2005d14749caa5fd9698d35f6f937664bde89cfdf0ab93675204c85bbf33`: `get_rgb` lines323–329 keeps `camera_data['rgba'][:, :, :3]`. Its upstream helper lines334–337 reads SAPIEN `Color`, scales255, clips and casts uint8. No channel permutation appears in these steps.
- `envs/utils/pkl2hdf5.py`, SHA `e119b4093c25f881c535a041bf1895e182ba00564d6ca57f6d6fe4fb62b2666c`: `images_encoding` lines10–22 calls `cv2.imencode('.jpg', imgs[i])` directly, stores `encoded_image.tobytes()`, and returns those encoded bytes with their maximum length. It performs no explicit channel swap.

Exact function text and full-file hashes are retained in `CAMERA-JPEG-SOURCE.json`. The relevant source path preserves camera channels0/1/2 into the OpenCV encoder. This does not establish that a PIL JPEG→RGB reader reproduces those native input bytes: that must be checked against the author encoding path in the same eventual CPU test. A PIL-written fixture alone does not cover this path. No image/array was loaded, source function executed, model/simulator imported, test/allocation created, scheduler queried or candidate changed for this handoff.
