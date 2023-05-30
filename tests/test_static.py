###############################################
# Tests for existence of files not in remote
###############################################
import os


def test_model_weights_exist():
    """Tests for existence of classifier model."""
    assert os.path.isfile("best_model.h5")


def test_heroku_setup():
    """Tests for existance of setup file with personal email"""
    assert os.path.isfile("setup.sh")
