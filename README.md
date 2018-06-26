
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

Unit tests are located in test directories throughout the project. The standard is to keep the test near the 
tested component, to make it easy to find the test when adapting the component.
