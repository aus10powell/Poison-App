###############################################
# Tests for existence of files not in remote
###############################################
import os

def test_model():
    """Tests for existence of classifier model.
    """
    assert os.path.isfile('best_model.h5')

def test_heroku_setup():
    """Tests for existance of setup file with personal email
    """
    assert os.path.isfile('setup.sh')