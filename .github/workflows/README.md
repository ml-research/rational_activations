
# Github Actions

## Self-hosted runner setup & statud
We are using a self-hosted runner on our servers that listens for jobs.
Should the runner stop running at any point (e.g. due to a server restart), we'll need to
make it run again as follows  (on the ml-student3 and 4 servers)

    cd actions-runner
    ./run.sh
    
To check whether the runner is active go to settings -> Actions -> scroll to the bottom, or click [here](https://github.com/ml-research/rational_activations/settings/actions)

## Tests

### When tests fail...


- CUDA out of memory: Please retry (commit & push) once enough memory is available.
    - Available servers are on the ml-student3 and ml-student4 servers, so if one is free you may redirect the CI to run on it, by
    replacing the runs-on commands in the [CI workflow file](compile_and_test.yml):
        ``runs-on: ml3`` or ``runs-on: ml4`` (instead of ``runs-on: self-hosted``)
 
 
 ## Linting
 
 We have right now an action that lints all python scripts within rational/ with flake8.
 ***Note that it does not stop the push if inconsistencies are found, but only prints them out.***