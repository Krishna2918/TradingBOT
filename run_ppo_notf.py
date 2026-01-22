"""Wrapper to run PPO training with TensorFlow blocked"""
import sys

# Block TensorFlow imports completely
class TensorFlowBlocker:
    def find_module(self, fullname, path=None):
        if 'tensorflow' in fullname or 'tensorboard' in fullname:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(f"TensorFlow imports are blocked: {fullname}")

sys.meta_path.insert(0, TensorFlowBlocker())

# Now run the training script
if __name__ == '__main__':
    import subprocess
    result = subprocess.run([sys.executable, 'train_ppo_no_tf_gpu.py'] + sys.argv[1:])
    sys.exit(result.returncode)
