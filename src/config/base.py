"""Base configuration system with automatic YAML-dataclass mapping."""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass, MISSING
from typing import Any, Dict, Optional, Type, TypeVar, get_type_hints, get_origin, get_args, Union

T = TypeVar('T')


def convert_value(value: Any, target_type: Type) -> Any:
    """
    Convert value to target type, handling special cases.

    Handles:
    - Path types
    - Optional types
    - Nested dataclasses
    - Basic types (int, float, str, bool)
    - Dict and List types
    """
    if value is None:
        return None

    # Get origin for generic types (Optional, List, Dict, etc.)
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Handle Union/Optional types
    if origin is Union:
        # For Optional[X], try to convert to X
        for arg in args:
            if arg != type(None):
                return convert_value(value, arg)
        return None

    # Handle Path
    if target_type == Path:
        return Path(value) if value is not None else None

    # Handle Dict types
    if origin is dict:
        if isinstance(value, dict) and args:
            # If we have type args like Dict[str, List[str]], convert values
            key_type, val_type = args[0], args[1] if len(args) > 1 else Any
            return {
                convert_value(k, key_type): convert_value(v, val_type)
                for k, v in value.items()
            }
        return value

    # Handle List types
    if origin is list:
        if isinstance(value, list) and args:
            item_type = args[0]
            return [convert_value(item, item_type) for item in value]
        return value

    # Handle nested dataclasses
    if is_dataclass(target_type):
        if isinstance(value, dict):
            return dataclass_from_dict(target_type, value)
        return value

    # Handle basic types
    if target_type in (int, float, str, bool):
        return target_type(value)

    return value


def dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Automatically create dataclass instance from dictionary.

    No manual field processing - uses dataclass metadata and type hints.
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    # Get type hints for proper type conversion
    type_hints = get_type_hints(cls)

    # Build kwargs by processing each field
    kwargs = {}
    for field_info in fields(cls):
        field_name = field_info.name
        field_type = type_hints.get(field_name, field_info.type)

        if field_name in data:
            # Convert and use provided value
            kwargs[field_name] = convert_value(data[field_name], field_type)
        elif field_info.default is not MISSING:
            # Use field default
            kwargs[field_name] = field_info.default
        elif field_info.default_factory is not MISSING:
            # Use default factory
            kwargs[field_name] = field_info.default_factory()
        else:
            # Check if Optional - if so, use None
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                if type(None) in args:
                    kwargs[field_name] = None
                else:
                    raise ValueError(f"Required field '{field_name}' missing for {cls.__name__}")
            else:
                raise ValueError(f"Required field '{field_name}' missing for {cls.__name__}")

    return cls(**kwargs)


def dataclass_to_dict(obj: Any, skip_none: bool = False) -> Any:
    """
    Automatically convert dataclass to dictionary.

    Handles nested dataclasses and Path objects.
    """
    if not is_dataclass(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    result = {}
    for field_info in fields(obj):
        value = getattr(obj, field_info.name)

        # Recursively convert nested structures
        if is_dataclass(value):
            value = dataclass_to_dict(value, skip_none)
        elif isinstance(value, Path):
            value = str(value)
        elif isinstance(value, dict):
            value = {k: dataclass_to_dict(v, skip_none) for k, v in value.items()}
        elif isinstance(value, list):
            value = [dataclass_to_dict(item, skip_none) for item in value]

        # Add to result if not None or if we're keeping None values
        if value is not None or not skip_none:
            result[field_info.name] = value

    return result


@dataclass
class AutoConfig:
    """Base class for automatic YAML configuration."""

    @classmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        """Automatically load configuration from YAML."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"To generate a template config, use:\n"
                f"  MainConfig.generate_template('{path}')"
            )

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        instance = dataclass_from_dict(cls, data)

        # Call validate if it exists
        if hasattr(instance, 'validate'):
            instance.validate()

        return instance

    def to_yaml(self, path: Path, skip_none: bool = True) -> None:
        """Automatically save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = dataclass_to_dict(self, skip_none=skip_none)

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

    @classmethod
    def generate_default(cls: Type[T], path: Path, overwrite: bool = False) -> None:
        """Generate default configuration file from dataclass defaults."""
        path = Path(path)

        if path.exists() and not overwrite:
            return

        # Create instance with defaults only
        instance = cls()
        instance.to_yaml(path, skip_none=False)