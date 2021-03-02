
# Github Actions

We are using a self-hosted runner on our servers that listens for jobs.
Should the runner stop running at any point (e.g. due to a server restart), we'll need to
make it run again as follows  (on the ml-student4 served)

    cd actions-runner
    ./run.sh
    
To check whether the runner is active go to settings -> Actions -> scroll to the bottom, or click [here](https://github.com/ml-research/rational_activations/settings/actions)