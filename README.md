
# Setup

To set up development environment:
cp config_dev.py config.py

To set up heavy gpu environment:
cp config_heavy_gpu.py config.py

To set up multi core environment:
cp config_multi.py config.py


configs can be inherited; 

``` 
from some_parent_config import * 
SOME_PROP = 'override value with this'
```

# Run

- run.py does a basic run
- profile.py does some basic profiling
- mini_profile.py does minimal profiling


# Test

Standard python unit tests are located in test directories throughout the project. 
For a file config.PROJECT_ROOT/some_path/some_file.py its test can be found at
config.PROJECT_ROOT/some_path/test/test_some_file.py
