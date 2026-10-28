"""CPU-only regression checks for cgroup-renumbered NVML versus global SLURM IDs."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("identity_probe", Path(__file__).resolve().parents[1] / "scripts/[train]gpu_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def physical(minor):
    return {"minor_number": minor, "uuid": f"GPU-{minor + 1:08x}-0000-0000-0000-{minor + 1:012x}",
            "pci_bus_id": f"0000:{80 + minor:02x}:00.0"}


class MappingTests(unittest.TestCase):
    def test_whole_node_needs_scheduler_ownership_and_all_physical_uuids(self):
        env = {"SLURM_JOB_ID": "12", "SLURM_STEP_ID": "0", "SLURMD_NODENAME": "gh001",
               "SLURM_JOB_GPUS": "0-3", "SLURM_STEP_GPUS": "0,1,2,3"}
        uuids = [physical(i)["uuid"] for i in range(4)]
        node = "NodeName=gh001 Gres=gpu:nvidia_gh200_120gb:4(S:0-3)"
        job = "JobId=12 JobState=RUNNING NumNodes=1 NodeList=gh001 AllocTRES=cpu=64,gres/gpu=4"
        probe.validate_whole_node_allocation(list(reversed(uuids)), uuids, env, node, job)
        for actual, environment, node_info, job_info in (
            (uuids[:2], env, node, job),
            (uuids[:-1] + [physical(5)["uuid"]], env, node, job),
            (uuids, dict(env, SLURM_STEP_GPUS="0-1"), node, job),
            (uuids, env, node, job.replace("gres/gpu=4", "gres/gpu=2")),
            (uuids, env, node, job.replace("NodeList=gh001", "NodeList=gh002")),
            (uuids, env, node, job.replace("RUNNING", "PENDING")),
            (uuids, env, node.replace(":4(", ":8("), job),
        ):
            with self.subTest(environment=environment, job=job_info), self.assertRaises(ValueError):
                probe.validate_whole_node_allocation(actual, uuids, environment, node_info, job_info)

    def test_configless_kernel_allocation_rejects_unconfined_or_wrong_devices(self):
        env = {"SLURM_JOB_ID": "12", "SLURM_STEP_ID": "0", "SLURM_JOB_GPUS": "2-3", "SLURM_STEP_GPUS": "2,3"}
        uuids = [physical(2)["uuid"], physical(3)["uuid"]]
        group = "0::/system.slice/slurmstepd.scope/job_12/step_0/user/task_0"
        probe.validate_kernel_allocation(uuids, uuids, env, group)
        for allowed, environment, cgroup in (
            ([physical(i)["uuid"] for i in range(4)], env, group),
            ([physical(0)["uuid"], physical(1)["uuid"]], env, group),
            (uuids, dict(env, SLURM_JOB_GPUS="0-3"), group),
            (uuids, env, group.replace("job_12", "job_13")),
            (uuids, env, group.replace("step_0", "step_1")),
        ):
            with self.subTest(allowed=allowed, environment=environment, cgroup=cgroup):
                with self.assertRaises(ValueError):
                    probe.validate_kernel_allocation(uuids, allowed, environment, cgroup)

    def test_faulted_card_does_not_hide_healthy_inventory_or_live_processes(self):
        fault = 'Unable to determine the device handle for GPU6: 0000:D2:00.0: Unknown Error\n'
        uuid = physical(0)['uuid']
        row = f'0, {uuid}, NVIDIA GeForce RTX 3090, 24576, 1, 0, 0000:50:00.0\n'
        with patch.object(probe.subprocess, 'run', return_value=probe.subprocess.CompletedProcess([], 6, row, fault)):
            inventory = probe.gpu_inventory()
        self.assertEqual([gpu['uuid'] for gpu in inventory], [uuid])
        with self.assertRaises(ValueError):
            probe.match_cuda_device_uuid(physical(6)['uuid'], inventory)
        with patch.object(probe.subprocess, 'run', return_value=probe.subprocess.CompletedProcess([], 6, fault + f'{uuid}, 123, python, 42\n', '')):
            self.assertEqual(probe.compute_processes()[0]['pid'], 123)
        with patch.object(probe.subprocess, 'run', return_value=probe.subprocess.CompletedProcess([], 6, fault, '')):
            with self.assertRaisesRegex(ValueError, 'no usable GPU'):
                probe.gpu_inventory()
        with patch.object(probe.subprocess, 'run', return_value=probe.subprocess.CompletedProcess([], 6, fault, 'Unexpected NVIDIA error')):
            with self.assertRaises(probe.subprocess.CalledProcessError):
                probe.gpu_inventory()

    def test_standalone_requires_explicit_host_full_uuid_and_no_scheduler(self):
        tokens = [physical(4)["uuid"]]
        probe.validate_direct_request({}, tokens, "research-host", "research-host", slurm_installed=False)
        for environment, selected, expected, actual, slurm in (
            ({}, tokens, "research-host", "different-host", False),
            ({}, tokens, "", "research-host", False),
            ({"SLURM_JOB_ID": "123"}, tokens, "research-host", "research-host", False),
            ({"SLURM_STEP_ID": "0"}, tokens, "research-host", "research-host", False),
            ({}, tokens, "research-host", "research-host", True),
            ({}, ["4"], "research-host", "research-host", False),
            ({}, ["GPU-0000"], "research-host", "research-host", False),
        ):
            with self.subTest(environment=environment, selected=selected, actual=actual, slurm=slurm):
                with self.assertRaises(ValueError):
                    probe.validate_direct_request(environment, selected, expected, actual, slurm_installed=slurm)

    def test_default_probe_still_rejects_login_shell_without_cuda_initialization(self):
        with patch.object(probe.sys, "platform", "linux"), patch.dict(probe.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "allocated srun step"):
                probe.run_probe(1)

    def test_node02_training_keeps_unmanaged_devices_outside_the_allocation(self):
        mapped = [{**physical(i), "index": i} for i in range(8)]
        unmanaged = [physical(8), physical(9)]
        reported = probe.check_inventory_for_purpose(mapped + unmanaged, mapped,
            purpose="training", node_names=("node02",))
        self.assertEqual(reported, unmanaged)
        probe.validate_allocation([row["uuid"] for row in mapped[:4]], mapped, {"SLURM_STEP_GPUS": "0-3"})
        with self.assertRaises(ValueError):
            probe.validate_allocation([row["uuid"] for row in unmanaged], mapped, {"SLURM_STEP_GPUS": "0-3"})
        with self.assertRaises(ValueError):
            probe.match_cuda_device_uuid(unmanaged[0]["uuid"], mapped)
        with self.assertRaisesRegex(RuntimeError, "excluded from measurements"):
            probe.check_inventory_for_purpose(mapped, mapped, purpose="measurement", node_names=("node02",))

    def test_training_purpose_still_rejects_identity_mismatches(self):
        mapped = [physical(0)]
        with self.assertRaisesRegex(ValueError, "PCI identity"):
            probe.check_inventory_for_purpose([{**mapped[0], "pci_bus_id": "0000:d3:00.0"}], mapped,
                purpose="training", node_names=("node02",))
        with self.assertRaises(ValueError):
            probe.check_inventory_for_purpose(mapped + [physical(8)], mapped,
                purpose="training", node_names=("node01",))

    def test_pro_renumbering_uses_global_gres_mapping(self):
        files = probe.expand_gres_gpu_files("Name=gpu Type=pro File=/dev/nvidia[0-3]\n", "node04")
        records = [physical(i) for i in range(4)]
        global_inventory = probe.build_global_gpu_inventory(files, records, {path:i for i,path in enumerate(files)})
        visible = [{**records[i], "index": i - 2, "pci_bus_id": f"00000000:{80 + i:02X}:00.0"} for i in (2, 3)]
        probe.crosscheck_physical_inventory(visible, global_inventory)
        uuids = [row["uuid"] for row in visible]
        environment = {"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_STEP_GPUS": "2,3", "SLURM_GPUS_ON_NODE": "2"}
        probe.validate_allocation(uuids, global_inventory, environment)
        # This is precisely the old bug: filtered NVML only has indices 0 and 1.
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            probe.validate_allocation(uuids, visible, environment)
        with self.assertRaisesRegex(ValueError, "SLURM_STEP_GPUS"):
            probe.validate_allocation([records[0]["uuid"], records[1]["uuid"]], global_inventory, environment)

    def test_node01_four_gpu_tail_and_job_subset(self):
        files = probe.expand_gres_gpu_files("Name=gpu Type=3090 File=/dev/nvidia[0-7]", "node01")
        records = [physical(i) for i in range(8)]
        global_inventory = probe.build_global_gpu_inventory(files, records, {path:i for i,path in enumerate(files)})
        uuids = [record["uuid"] for record in records[4:]]
        probe.validate_allocation(uuids, global_inventory, {"SLURM_STEP_GPUS": "4-7", "SLURM_JOB_GPUS": "0-7"})
        with self.assertRaisesRegex(ValueError, "SLURM_JOB_GPUS"):
            probe.validate_allocation(uuids, global_inventory, {"SLURM_STEP_GPUS": "4-7", "SLURM_JOB_GPUS": "0-6"})

    def test_explicit_step_subset_keeps_both_scheduler_boundaries(self):
        inventory = [{**physical(i), "index": i} for i in range(9)]
        selected = [inventory[i]["uuid"] for i in (0, 4, 5, 7)]
        environment = {"SLURM_STEP_GPUS": "0-8", "SLURM_JOB_GPUS": "0-8"}
        probe.validate_allocation(selected, inventory, environment, allow_step_subset=True)
        with self.assertRaisesRegex(ValueError, "SLURM_STEP_GPUS"):
            probe.validate_allocation(selected, inventory, environment)
        for field in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                probe.validate_allocation(selected, inventory, {**environment, field: "0-6"},
                                          allow_step_subset=True)
        with patch.object(probe.sys, "platform", "linux"):
            with self.assertRaisesRegex(ValueError, "not a timing measurement"):
                probe.run_probe(4, allow_step_subset=True)

    def test_device_minor_is_read_from_character_device_not_inferred_from_name(self):
        files = ["/dev/nvidia0", "/dev/nvidia1"]
        records = [physical(7), physical(9)]
        mapping = probe.build_global_gpu_inventory(files, records, {files[0]:7, files[1]:9})
        self.assertEqual(mapping[0]["minor_number"], 7)
        self.assertEqual(mapping[1]["uuid"], physical(9)["uuid"])
        self.assertEqual(mapping[1]["slurm_global_index"], 1)
        probe.validate_allocation([physical(9)["uuid"]], mapping, {"SLURM_STEP_GPUS": "1"})
        with self.assertRaises(ValueError):
            probe.build_global_gpu_inventory(files, records, {files[0]:7, files[1]:7})
        with self.assertRaises(ValueError):
            probe.build_global_gpu_inventory(files, records, {files[0]:7, files[1]:8})

    def test_gres_bitmap_indexes_follow_configured_order_not_device_suffixes(self):
        files = probe.expand_gres_gpu_files("NodeName=node09 Name=gpu Type=A100 File=/dev/nvidia[3-7]", "node09")
        records = [physical(i) for i in range(8)]
        mapping = probe.build_global_gpu_inventory(files, records,
            {path: int(path.removeprefix("/dev/nvidia")) for path in files})
        selected = [physical(i)["uuid"] for i in range(3, 8)]
        probe.validate_allocation(selected, mapping, {"SLURM_STEP_GPUS": "0,1,2,3,4"})
        with self.assertRaises(ValueError):
            probe.validate_allocation(selected, mapping, {"SLURM_STEP_GPUS": "3,4,5,6,7"})

    def test_removed_middle_device_preserves_scheduler_slot_to_physical_uuid_mapping(self):
        files = probe.expand_gres_gpu_files("Name=gpu Type=3090 File=/dev/nvidia[0-3,5-7]", "node01")
        records = [physical(i) for i in range(8)]
        mapping = probe.build_global_gpu_inventory(files, records,
            {path: int(path.removeprefix("/dev/nvidia")) for path in files})
        probe.validate_allocation([physical(7)["uuid"]], mapping,
                                  {"SLURM_STEP_GPUS": "6", "SLURM_JOB_GPUS": "6"})
        self.assertEqual(mapping[6]["device_file"], "/dev/nvidia7")
        with self.assertRaisesRegex(ValueError, "SLURM_STEP_GPUS"):
            probe.validate_allocation([physical(6)["uuid"]], mapping, {"SLURM_STEP_GPUS": "6"})

    def test_proc_uuid_and_pci_identity_crosscheck(self):
        record = probe.parse_proc_gpu_information("GPU UUID: GPU-00000001-0000-0000-0000-000000000001\nDevice Minor: 0\nBus Location: 0000:d2:00.0\n")
        self.assertEqual(record["pci_bus_id"], "0000:d2:00.0")
        self.assertEqual(probe.canonical_pci_bus_id("00000000:D2:00.0"), "0000:d2:00.0")
        with self.assertRaisesRegex(ValueError, "PCI identity"):
            probe.crosscheck_physical_inventory([{**record, "pci_bus_id": "0000:d3:00.0"}], [record])

    def test_ambiguous_or_unverified_layouts_fail_closed(self):
        for text in ("AutoDetect=nvml", "Name=gpu Count=2", "Name=gpu File=/dev/nvidia[2-1]",
                     "Name=gpu File=/dev/nvidia[0,0]", "Name=gpu File=/dev/nvidia[0-1] Count=3",
                     "Name=gpu File=/dev/nvidia0 Flags=CountOnly"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                probe.expand_gres_gpu_files(text, "node01")
        with self.assertRaisesRegex(ValueError, "Missing SLURM"):
            probe.validate_allocation([physical(0)["uuid"]], [{**physical(0), "index":0}], {})


if __name__ == "__main__":
    unittest.main()
