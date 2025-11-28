"""
GPU Device Manager
Handles GPU device selection and configuration gracefully
"""

import os
import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class GPUDeviceManager:
    """Manages GPU device selection and configuration"""

    def __init__(self):
        self._available_gpus = None
        self._selected_devices = None

    def get_available_gpus(self) -> List[int]:
        """
        Get list of available GPU device IDs

        Returns:
            List of GPU IDs (e.g., [0, 1, 2, 3])
        """
        if self._available_gpus is not None:
            return self._available_gpus

        try:
            import torch
            if torch.cuda.is_available():
                self._available_gpus = list(range(torch.cuda.device_count()))
                logger.info(f"Found {len(self._available_gpus)} GPUs: {self._available_gpus}")
            else:
                self._available_gpus = []
                logger.warning("CUDA not available, no GPUs detected")
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
            self._available_gpus = []

        return self._available_gpus

    def get_gpu_memory_info(self, device_id: int) -> dict:
        """
        Get memory information for a specific GPU

        Args:
            device_id: GPU device ID

        Returns:
            Dict with keys: total_gb, used_gb, free_gb
        """
        try:
            # Try pynvml first (most accurate)
            try:
                import pynvml
                if not hasattr(self, '_nvml_initialized'):
                    pynvml.nvmlInit()
                    self._nvml_initialized = True

                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                return {
                    'total_gb': mem_info.total / 1024**3,
                    'used_gb': mem_info.used / 1024**3,
                    'free_gb': mem_info.free / 1024**3,
                }
            except (ImportError, Exception):
                # Fallback to torch
                import torch
                props = torch.cuda.get_device_properties(device_id)
                total_gb = props.total_memory / 1024**3

                # Get currently allocated memory (not perfect but best we can do)
                torch.cuda.set_device(device_id)
                mem_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(device_id) / 1024**3

                # Estimate free memory
                used_gb = mem_reserved if mem_reserved > 0 else mem_allocated
                free_gb = total_gb - used_gb

                return {
                    'total_gb': total_gb,
                    'used_gb': used_gb,
                    'free_gb': free_gb,
                }
        except Exception as e:
            logger.warning(f"Failed to get memory info for GPU {device_id}: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0}

    def get_gpu_with_most_free_memory(self) -> Optional[int]:
        """
        Find GPU with the most free memory

        Returns:
            GPU ID with most free memory, or None if no GPUs available
        """
        available_gpus = self.get_available_gpus()
        if not available_gpus:
            return None

        best_gpu = None
        max_free_memory = -1

        for gpu_id in available_gpus:
            mem_info = self.get_gpu_memory_info(gpu_id)
            free_gb = mem_info['free_gb']

            logger.info(f"GPU {gpu_id}: {free_gb:.2f} GB free / {mem_info['total_gb']:.2f} GB total")

            if free_gb > max_free_memory:
                max_free_memory = free_gb
                best_gpu = gpu_id

        if best_gpu is not None:
            logger.info(f"GPU {best_gpu} has most free memory: {max_free_memory:.2f} GB")

        return best_gpu

    def select_devices(
        self,
        devices: Optional[Union[int, str, List[int]]] = None,
        auto_select: bool = True,
        set_env: bool = True
    ) -> List[int]:
        """
        Select GPU devices to use

        Args:
            devices: Device specification:
                - None: Auto-select first available GPU
                - int: Single GPU ID (e.g., 0)
                - str: Comma-separated GPU IDs (e.g., "0,1") or "all"
                - List[int]: List of GPU IDs (e.g., [0, 1])
            auto_select: If True and devices is None, auto-select first GPU
            set_env: If True, set CUDA_VISIBLE_DEVICES environment variable

        Returns:
            List of selected GPU IDs

        Examples:
            >>> manager = GPUDeviceManager()
            >>> manager.select_devices(0)  # Use GPU 0
            [0]
            >>> manager.select_devices("0,1")  # Use GPUs 0 and 1
            [0, 1]
            >>> manager.select_devices("all")  # Use all GPUs
            [0, 1, 2, 3]
            >>> manager.select_devices(None)  # Auto-select first GPU
            [0]
        """
        available_gpus = self.get_available_gpus()

        if not available_gpus:
            logger.warning("No GPUs available, running on CPU")
            self._selected_devices = []
            return []

        # Parse device specification
        if devices is None:
            if auto_select:
                # Auto-select GPU with most free memory
                best_gpu = self.get_gpu_with_most_free_memory()
                if best_gpu is not None:
                    selected = [best_gpu]
                    logger.info(f"Auto-selected GPU {best_gpu} (most free memory)")
                else:
                    selected = []
                    logger.warning("No GPU with available memory found")
            else:
                selected = []
        elif isinstance(devices, int):
            # Single GPU ID
            if devices in available_gpus:
                selected = [devices]
            else:
                raise ValueError(
                    f"GPU {devices} not available. Available GPUs: {available_gpus}"
                )
        elif isinstance(devices, str):
            if devices.lower() == "all":
                # Use all available GPUs
                selected = available_gpus
                logger.info(f"Selected all GPUs: {selected}")
            else:
                # Parse comma-separated string
                try:
                    selected = [int(d.strip()) for d in devices.split(",")]
                    # Validate all devices are available
                    for device_id in selected:
                        if device_id not in available_gpus:
                            raise ValueError(
                                f"GPU {device_id} not available. Available GPUs: {available_gpus}"
                            )
                except ValueError as e:
                    raise ValueError(
                        f"Invalid device specification '{devices}': {e}"
                    )
        elif isinstance(devices, list):
            # List of GPU IDs
            selected = devices
            for device_id in selected:
                if device_id not in available_gpus:
                    raise ValueError(
                        f"GPU {device_id} not available. Available GPUs: {available_gpus}"
                    )
        else:
            raise TypeError(
                f"Invalid devices type: {type(devices)}. "
                "Expected int, str, list, or None"
            )

        self._selected_devices = selected

        # Set environment variable
        if set_env and selected:
            devices_str = ",".join(map(str, selected))
            os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
            logger.info(f"Set CUDA_VISIBLE_DEVICES={devices_str}")

        return selected

    def get_selected_devices(self) -> List[int]:
        """Get currently selected devices"""
        if self._selected_devices is None:
            return self.select_devices()
        return self._selected_devices

    def print_gpu_info(self):
        """Print detailed GPU information including free memory"""
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA is not available")
                return

            print("\n" + "="*70)
            print("GPU Information")
            print("="*70)

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_info = self.get_gpu_memory_info(i)

                print(f"\nGPU {i}: {props.name}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {mem_info['total_gb']:.2f} GB")
                print(f"  Used Memory:  {mem_info['used_gb']:.2f} GB")
                print(f"  Free Memory:  {mem_info['free_gb']:.2f} GB")
                print(f"  Multi-Processors: {props.multi_processor_count}")

                # Show percentage
                if mem_info['total_gb'] > 0:
                    usage_pct = (mem_info['used_gb'] / mem_info['total_gb']) * 100
                    print(f"  Memory Usage: {usage_pct:.1f}%")

            print("\n" + "="*70)

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")


# Global instance
_device_manager = GPUDeviceManager()


# Convenience functions
def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs"""
    return _device_manager.get_available_gpus()


def select_devices(
    devices: Optional[Union[int, str, List[int]]] = None,
    auto_select: bool = True,
    set_env: bool = True
) -> List[int]:
    """Select GPU devices to use (see GPUDeviceManager.select_devices)"""
    return _device_manager.select_devices(devices, auto_select, set_env)


def get_selected_devices() -> List[int]:
    """Get currently selected devices"""
    return _device_manager.get_selected_devices()


def print_gpu_info():
    """Print detailed GPU information"""
    _device_manager.print_gpu_info()


# Command-line interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Device Manager")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="GPU devices to use (e.g., '0', '0,1', 'all')"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print GPU information"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.info:
        print_gpu_info()
    else:
        selected = select_devices(args.devices)
        print(f"\nSelected GPUs: {selected}")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
