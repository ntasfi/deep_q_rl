# PLE DQN Benchmark Fork

This is a fork of Nathan Sprague's DQN implementation. It is setup to run benchmarks on various PLE games. All the good stuff is Nathan's, bugs and whatnot are mine!

Please read the text below to install.

**This has only been run on Ubuntu 14.04**

# Dependencies

* GPU
* OpenCV (will replace with scikit.image later)
* Theano
* Lasagne
* PyGame Learning Environment

The script `dep_script.sh` can be used to install all dependencies under Ubuntu. Should work, havent tested. Use at your own risk.

# Running

Use the scripts `run_nips.py` or `run_nature.py` to start all the necessary processes:

`$ ./run_nips.py -g "Pixelcopter"`

`$ ./run_nature.py --g "Pixelcopter"`

The `run_nips.py` script uses parameters consistent with the original
NIPS workshop paper.  This code should take 2-4 days to complete.  

Either script will store output files in a folder prefixed with the
name of the game.  Pickled version of the network objects are stored
after every epoch.  The file `results.csv` will contain the testing
output.  You can plot the progress by executing `plot_results.py`:

`$ python plot_results.py pixelcopter_05-28-17-09_0p00025_0p99/results.csv`

After training completes, you can watch the network play using the 
`ple_run_watch.py` script: 

`$ python run_watch.py pixelcopter_05-28-17-09_0p00025_0p99/network_file_99.pkl`
