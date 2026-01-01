#!/usr/bin/env python3
"""
Convert HDF5 dataset feature groups from old naming to new naming.

Old naming:
- /features/ic/ → /features/initial/
- /features/nop/ → /features/architecture/
- /features/sdf/ → /features/summary/
- /features/td/ → /features/temporal/

Usage:
    python scripts/dev/convert_h5_feature_naming.py datasets/my_dataset.h5 [--dry-run] [--backup]

Options:
    --dry-run: Show what would be changed without modifying the file
    --backup: Create a backup copy before modifying (recommended)
"""

import h5py
import shutil
import argparse
from pathlib import Path
from typing import Dict


# Mapping from old names to new names
FAMILY_RENAME_MAP = {
    "ic": "initial",
    "nop": "architecture",
    "sdf": "summary",
    "td": "temporal"
}


def convert_h5_feature_naming(
    dataset_path: Path,
    dry_run: bool = False,
    backup: bool = False
) -> None:
    """
    Convert feature family naming in HDF5 dataset.

    Args:
        dataset_path: Path to HDF5 dataset
        dry_run: If True, only print changes without modifying
        backup: If True, create backup before modifying
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Create backup if requested
    if backup and not dry_run:
        backup_path = dataset_path.with_suffix(dataset_path.suffix + ".backup")
        print(f"Creating backup: {backup_path}")
        shutil.copy2(dataset_path, backup_path)

    # Open dataset
    mode = 'r' if dry_run else 'r+'
    with h5py.File(dataset_path, mode) as f:
        # Check if features group exists
        if 'features' not in f:
            print("No /features/ group found in dataset")
            return

        features_group = f['features']
        families_found = list(features_group.keys())
        print(f"\nFeature families found: {families_found}")

        # Find families to rename
        families_to_rename = {}
        for old_name, new_name in FAMILY_RENAME_MAP.items():
            if old_name in families_found:
                families_to_rename[old_name] = new_name

        if not families_to_rename:
            print("\nNo feature families need renaming (already using new names or not present)")
            return

        print(f"\nFamilies to rename: {families_to_rename}")

        if dry_run:
            print("\n[DRY RUN] Would perform the following renames:")
            for old_name, new_name in families_to_rename.items():
                print(f"  /features/{old_name}/ → /features/{new_name}/")
            print("\nRe-run without --dry-run to apply changes")
            return

        # Perform renames
        print("\nRenaming groups...")
        for old_name, new_name in families_to_rename.items():
            old_path = f"/features/{old_name}"
            new_path = f"/features/{new_name}"

            print(f"  {old_path} → {new_path}")

            # HDF5 doesn't have native rename, so we copy and delete
            f.copy(old_path, new_path)
            del f[old_path]

        # Update @family_versions attribute if it exists
        if 'family_versions' in features_group.attrs:
            family_versions = dict(features_group.attrs['family_versions'])
            print(f"\nOriginal family_versions: {family_versions}")

            # Update keys
            updated_versions = {}
            for old_name, version in family_versions.items():
                new_name = FAMILY_RENAME_MAP.get(old_name, old_name)
                updated_versions[new_name] = version

            # Write back
            del features_group.attrs['family_versions']
            features_group.attrs['family_versions'] = updated_versions
            print(f"Updated family_versions: {updated_versions}")

        print("\n✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 dataset feature groups from old naming to new naming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (recommended first step)
  python scripts/dev/convert_h5_feature_naming.py datasets/my_data.h5 --dry-run

  # Apply changes with backup
  python scripts/dev/convert_h5_feature_naming.py datasets/my_data.h5 --backup

  # Apply changes without backup (not recommended)
  python scripts/dev/convert_h5_feature_naming.py datasets/my_data.h5
        """
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to HDF5 dataset file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup copy before modifying (recommended)"
    )

    args = parser.parse_args()

    try:
        convert_h5_feature_naming(
            dataset_path=args.dataset,
            dry_run=args.dry_run,
            backup=args.backup
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
