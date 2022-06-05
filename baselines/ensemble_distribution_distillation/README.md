## Importing from parent dirs when running scripts in subdirs
As per this [guide](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html) this is simply not possible without modifying the python path.

This is easily done with this command:
```
# If you are in a virtual env, then:
# VIRTUAL_ENV is <path_to_project>/venv
export PYTHONPATH=${PYTHONPATH}:${VIRTUAL_ENV}/../src
```

This disappears when a terminal session is shut down, but this can be made permanent by adding the above line to your
virtual environment `activate` script.
