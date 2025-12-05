"""Memory monitoring utilities for debugging memory leaks."""

from __future__ import annotations

import gc
import psutil
import torch


def get_memory_stats() -> dict:
    """
    Get current memory usage statistics.

    Pure function: Only reads current system state.
    """
    stats = {}

    # CPU Memory (process-specific)
    process = psutil.Process()
    memory_info = process.memory_info()
    stats['cpu_rss_mb'] = memory_info.rss / 1024 / 1024  # Resident Set Size
    stats['cpu_vms_mb'] = memory_info.vms / 1024 / 1024  # Virtual Memory Size

    # System Memory
    sys_memory = psutil.virtual_memory()
    stats['system_available_mb'] = sys_memory.available / 1024 / 1024
    stats['system_used_percent'] = sys_memory.percent

    # GPU Memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all operations complete
        stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Get GPU device properties
        props = torch.cuda.get_device_properties(0)
        stats['gpu_total_mb'] = props.total_memory / 1024 / 1024
        stats['gpu_free_mb'] = stats['gpu_total_mb'] - stats['gpu_reserved_mb']
    else:
        stats['gpu_allocated_mb'] = 0
        stats['gpu_reserved_mb'] = 0
        stats['gpu_max_allocated_mb'] = 0
        stats['gpu_total_mb'] = 0
        stats['gpu_free_mb'] = 0

    return stats


def print_memory_summary(label: str, stats: dict = None) -> None:
    """
    Print formatted memory summary.

    Single responsibility: Only handles memory stats display.
    """
    if stats is None:
        stats = get_memory_stats()

    print(f"\n=== MEMORY [{label}] ===")
    print(f"CPU: {stats['cpu_rss_mb']:.1f}MB RSS, {stats['cpu_vms_mb']:.1f}MB VMS")
    print(f"System: {stats['system_used_percent']:.1f}% used, {stats['system_available_mb']:.0f}MB available")
    print(f"GPU: {stats['gpu_allocated_mb']:.1f}MB allocated, {stats['gpu_reserved_mb']:.1f}MB reserved")
    print(f"GPU: {stats['gpu_free_mb']:.1f}MB free / {stats['gpu_total_mb']:.0f}MB total")


def force_garbage_collection() -> None:
    """
    Force garbage collection and GPU cache cleanup.

    Single responsibility: Only handles memory cleanup.
    """
    # Force Python garbage collection
    collected = gc.collect()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if collected > 0:
        print(f"DEBUG: Garbage collected {collected} objects")


def track_memory_usage(func):
    """
    Decorator to track memory usage before/after function execution.

    Use for debugging memory leaks in specific functions.
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        # Memory before
        stats_before = get_memory_stats()
        print_memory_summary(f"BEFORE {func_name}", stats_before)

        # Execute function
        result = func(*args, **kwargs)

        # Memory after
        stats_after = get_memory_stats()
        print_memory_summary(f"AFTER {func_name}", stats_after)

        # Calculate differences
        cpu_diff = stats_after['cpu_rss_mb'] - stats_before['cpu_rss_mb']
        gpu_diff = stats_after['gpu_allocated_mb'] - stats_before['gpu_allocated_mb']

        print(f"DELTA: CPU +{cpu_diff:.1f}MB, GPU +{gpu_diff:.1f}MB")

        return result

    return wrapper