# AnyViewPhys cold archive with32GiB parts

R2 single-object BOS upload failed with EIO after167,805,714,432 bytes (10002 writes of16MiB); no source bytes were evicted and the complete249,659,603,127-byte local archive is intact. This offset is consistent with the [documented10000 multipart part-number bound](https://cloud.baidu.com/doc/BOS/s/tl981rdoy); attributing the mount error to that bound is an inference. Do not repeat the same large-object upload.

Use32GiB objects with per-part SHA256 and independent readback into fresh staging. Match whole-archive SHA256 after ordered concatenation, extract the readback, compare every restored member and unchanged source metadata/ctime, then use fresh scheduler/readers and the qualified quarantine/release. The fixture forces128-byte objects to qualify multiple split boundaries. Keep unrelated jobs running.
