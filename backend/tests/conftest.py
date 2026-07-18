import sys
import os
import pytest

# Add backend directory to python path first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch

# Mock the heavy pre-training steps before importing any backend modules
patcher_train = patch("entity_interface.live_entity.LiveEntity._train_cifn_classifier")
patcher_step = patch("entity_interface.live_entity.LiveEntity._run_internal_training_step")

patcher_train.start()
patcher_step.start()

@pytest.fixture
def anyio_backend():
    return 'asyncio'
