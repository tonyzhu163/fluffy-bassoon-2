"""Attest Baige container GPUs by PCI identity when NVML rewrites their UUIDs.

The scheduler's NVIDIA_VISIBLE_DEVICES remains authoritative. CUDA ordinals
are joined to that allocation by the driver's PCI address, never by order.
"""
import ctypes
import json
import os
from pathlib import Path
import re
import socket


def validate_container(environ, hostname):
    if (not re.fullmatch(r'job-[a-z0-9]+-master-0', hostname)
            or not environ.get('KUBERNETES_SERVICE_HOST')
            or any(environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_STEP_ID'))):
        raise ValueError('Baige mapping requires a scheduled Baige master container')
    assigned = environ.get('NVIDIA_VISIBLE_DEVICES', '').split(',')
    if not assigned or len(set(assigned)) != len(assigned) or any(
            not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', x)
            for x in assigned):
        raise ValueError('Baige requires an explicit scheduler NVIDIA UUID allocation')
    return assigned


def pci(value):
    match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value.strip())
    if not match:
        raise ValueError('Invalid GPU PCI address')
    domain, bus, device, function = match.groups()
    return f'{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}'


def cuda_pci(ordinal):
    driver = ctypes.CDLL('libcuda.so.1')
    if driver.cuInit(0):
        raise RuntimeError('CUDA driver initialization failed')
    device = ctypes.c_int()
    if driver.cuDeviceGet(ctypes.byref(device), ordinal):
        raise RuntimeError('Cannot resolve visible CUDA ordinal')
    address = ctypes.create_string_buffer(64)
    if driver.cuDeviceGetPCIBusId(address, len(address), device):
        raise RuntimeError('Cannot read CUDA PCI identity')
    return pci(address.value.decode())


def map_devices(cuda_devices, nvml_devices, assigned):
    if (len(cuda_devices) != len(nvml_devices) or len(assigned) != len(nvml_devices)
            or set(assigned) != {x['uuid'] for x in nvml_devices}):
        raise ValueError('CUDA/NVML inventory differs from the scheduler allocation')
    if (len({pci(x['pci_bus_id']) for x in nvml_devices}) != len(nvml_devices)
            or len({pci(x['pci_bus_id']) for x in cuda_devices}) != len(cuda_devices)
            or len({x['cuda_uuid'] for x in cuda_devices}) != len(cuda_devices)):
        raise ValueError('GPU identities must be unique')
    mapped = []
    for row in cuda_devices:
        matches = [x for x in nvml_devices if pci(x['pci_bus_id']) == pci(row['pci_bus_id'])]
        if len(matches) != 1 or row['cuda_name'] != matches[0]['name']:
            raise ValueError('CUDA device has no unique matching allocated NVML PCI identity')
        mapped.append(dict(row, uuid=matches[0]['uuid'], inventory=matches[0]))
    return mapped


def mapping():
    validate_container(os.environ, socket.gethostname())
    report = json.loads(Path(os.environ['LYCO_BAIGE_GPU_MAP']).read_text())
    if (report['status'] != 'passed' or report['allocation_type'] != 'baige_container'
            or report['hostname'] != socket.gethostname()):
        raise ValueError('GPU mapping belongs to a different or unqualified allocation')
    rows = report['devices']
    map_devices(rows, [x['inventory'] for x in rows], validate_container(os.environ, socket.gethostname()))
    return rows


def visibility_for(nvml_uuid):
    matches = [x for x in mapping() if x['uuid'] == nvml_uuid]
    if len(matches) != 1:
        raise ValueError('Evaluator GPU is outside the attested allocation')
    return matches[0]['visibility_token']


def selected_device():
    selected = os.environ.get('LYCO_BAIGE_ISOLATED_UUID')
    matches = [x for x in mapping() if (x['uuid'] == selected if selected else
               x['visibility_token'] == os.environ.get('CUDA_VISIBLE_DEVICES'))]
    if len(matches) != 1:
        raise ValueError('Require exactly one attested Baige GPU selection')
    return matches[0]


def verify_selected_cuda(torch):
    row = selected_device()
    if (torch.cuda.device_count() != 1 or cuda_pci(0) != pci(row['pci_bus_id'])
            or str(torch.cuda.get_device_properties(0).uuid) != row['cuda_uuid']):
        raise RuntimeError('Isolated CUDA device differs from the attested GPU')
    return row['uuid']
