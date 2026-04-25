"""
Checkpoint Manager using Orbax
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Any, Tuple, List
import orbax.checkpoint as ocp
import jax

class CheckpointManager:
    """
    Manages saving and restoring checkpoints using Orbax.
    Supports 'keep_last_n' and auto-resume.
    """
    def __init__(self, directory: str, keep_last_n: int = 5):
        self.directory = Path(directory).absolute()
        self.keep_last_n = keep_last_n
        
        # Configure Orbax Checkpointer
        # Use StandardCheckpointer for PyTrees
        self.checkpointer = ocp.StandardCheckpointer()
        
        # Use Orbax's built-in CheckpointManager for file management
        # It handles atomic saves and keeping last N
        options = ocp.CheckpointManagerOptions(max_to_keep=keep_last_n if keep_last_n > 0 else None, create=True)
        self.manager = ocp.CheckpointManager(
            self.directory, 
            self.checkpointer, 
            options=options
        )

    def save(self, step: int, item: Any, metrics: Optional[dict] = None):
        """Save a checkpoint at the given step."""
        try:
            save_args = ocp.args.StandardSave(item)
            self.manager.save(step, args=save_args, metrics=metrics)
        except (PermissionError, OSError) as e:
            # WSL + NTFS 偶发权限问题，跳过本次保存而非崩溃
            print(f"[-] Checkpoint save failed (step={step}), skipping: {e}")
        except Exception as e:
            print(f"[-] Checkpoint save unexpected error (step={step}), skipping: {e}")

    def restore_latest(self, item_structure: Any = None) -> Tuple[Optional[Any], int]:
        """
        Restore the latest checkpoint.
        
        Args:
            item_structure: The structure of the item (e.g. TrainState) to restore into.
                            If None, restores as a raw dictionary/PyTree.
                            Providing structure is efficient for shape info.
                            
        Returns:
            (restored_item, step) if found, else (None, 0)
        """
        latest_step = self.manager.latest_step()
        if latest_step is None:
            return None, 0
            
        print(f"🔄 Restoring checkpoint from step {latest_step}...")
        
        # Create RestoreArgs if structure is provided (helps with array sharding/layouts if needed)
        # For simple use cases, standard restore is enough.
        
        try:
            if item_structure is not None:
                 restore_args = ocp.args.StandardRestore(item_structure)
                 restored = self.manager.restore(latest_step, args=restore_args)
            else:
                 restored = self.manager.restore(latest_step)
                 
            return restored, latest_step
        except Exception as e:
            print(f"❌ Failed to restore checkpoint: {e}")
            return None, 0

    def list_checkpoints(self) -> List[int]:
        return self.manager.all_steps()

    def wait_until_finished(self):
        """Block until all checkpoint operations are completed."""
        self.manager.wait_until_finished()
