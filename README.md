# DQN and Double DQN on Atari games

<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/pierclgr/Atari-DQN/main/test_videos/Breakout_DQN.gif" width="160" height="210" alt="Breakout DQN">
<img src="https://raw.githubusercontent.com/pierclgr/Atari-DQN/main/test_videos/Breakout_DoubleDQN.gif" width="160" height="210" alt="Breakout DoubleDQN">
</div>

## Abstract 

The following project is a Python implementation of DQN and Double DQN algorithms and a comparison between the 
performances of the twos in the videogame Breakout for Atari 2600. The scope of this project is thus to compare the two 
algorithms showing the differences in performance and value estimates, along with the effects of the improvements that 
Double DQN introduced over DQN.
For more details, it is possible to check out the [project report](https://github.com/pierclgr/Atari-DQN/blob/main/report/report.pdf).

## Installation on a local machine or a server
In order to install the repository and run training/experiments on your local machine, you first have to set up the 
python environment: 
1. Clone the repository using `git` with the command
```shell
git clone https://github.com/pierclgr/Atari-DQN
```
2. Open a terminal in the repository directory
3. Create a Python 3.7.13 environment and install the packages using the file `requirements.txt` using `pip` and the command
```shell
pip install -r requirements.txt
```

Once you do this, you're set to go and you have a Python environment with all the required packages.

### Running experiments
Each experiment is associated with a Hydra `.yaml` configuration file. If you need more infos about hydra and its 
configuration files, you can look at the official [hydra documentation](https://hydra.cc/docs/intro/). In order to set 
up any training/testing experiment, you need to create a configuration file for hydra or to modify the already existing
configuration files that are existing in the `config` folder.

#### Training
To run a training experiment, you need to create a configuration file that is similar to one of the two training
configuration files in the `config` folder. We are giving our configuration files in this folder and they are named 
`breakout_train_dqn.yaml`, which is the configuration that we used to train DQN, and `breakout_train_doubledqn.yaml`, 
which is the configuration file that we used to train Double DQN. You can change all the parameters you want there, but
to reproduce our results we suggest to maintain everything the same.

In order to run a training experiment, you have to do some changes to the configuration file:
1. Set up the logging
   1. If you want to log the metrics on wandb, you have to 
      1. Create a Wandb account
      2. After creating a Wandb account, create a project
      3. Change the `wandb` field in the configuration file, by changing its subfield `project_name` to the name of you project in Wandb and its `entity_name` subfield to your username on Wandb
      4. _\*optional\*_: you can also change the subfield `run_name` to change the name of the run on Wandb, this will be the prefix of the run name on Wandb that will be concatenated with a randomly generated suffix
   2. Otherwise, if you don't want to log the metrics, you can just set the `logging` field of the configuration file to `false`, metrics will not be logged and will just be outputted to the console
2. Change the `home_directory` field of the configuration file to the path of you repository directory; it's very important that the path you put is an _**absolute**_ path
3. Make sure that the configuration file you want to use is in the folder `config` that is in the root directory of your repository

Afer you do this, you have to:
1. Open a terminal in the repository directory
2. Activate the environment created before
3. Launch the following command by specifiying your training configuration file name (without the extension) in the argument `--config-name`
```shell
python src/trainer.py --config-name=your_training_configuration_file_name
```

If you did everything correctly, the training will start. If you set up logging with Wandb, at the beginning of you first run with logging you will be required to specify if to consider an existing Wandb project for 
logging or create a new one. We suggest to use the already created project. Moreover, you will be asked to insert an API key that is available on your Wandb account, however the process is guided so you just have to follow 
the instructions that are outputted in the console.

After you complete a training successfully, the program will save the model weights into a `.pt` file in the folder `trained_models` in the root directory of the repository. You can change the name of the output file by modifying the field
`output_model_file` in the configuration file.

#### Testing
You can use a file saved after the training to test the agent and watch it play. In order to test the trained agent, you have to configure a configuration file that is similar to the one for training. Again, we are providing also two files, 
one to test DQN and one to test Double DQN, that are called `breakout_train_dqn.yaml` and `breakout_train_doubledqn.yaml` respectively. You thus need to do some changes to the testing configuration file:
1. Change the Wandb logging field the same way you did with training if you want to use logging, otherwise just disable it as you did in the training configuration
2. Change the field `output_model_file` if you changed the name of the output file during training
3. Make sure that the output model file that you want to use is in the folder `trained_models` in the root directory of the repository
4. Make sure that the testing configuration file that you want to use is in the `config` folder in the root directory of the repository

After you do this, you have to:
1. Open a terminal in the repository directory
2. Activate the environment created before
3. Launch the following command by specifying your testing configuration file name (without the extension) in the argument `--config-name`
```shell
python src/tester.py --config-name=your_testing_configuration_file_name
```

If you did everything correctly, the testing will start and you will see the agent playing on you screen. Hit `CTRL+C` in your terminal when you want to stop playing.

## Installation on Google Colab
If you wish to run training or testing on Google Colab, you can just open the `colab_notebook.ipynb` notebook in Colab by clicking the button in the top of the notebook. 
After you did this, execute the first code cell: it will clone the repository and install the required libraries.

#### Training
Follow the steps that are shown in the Training section related to installation on servers or local machines. We also suggest to train for less than 12 hours in order to avoid Colab crashes or timeouts and keep the output model saved in Colab's memory. As an alternative, you can mount you Google Drive and save 
the trained model there also for future testings or trainings. If you use a different configuration file name, change the code in the cell block by changing the `--config-name` argument to the name of your training configuration file. Make sure to use GPU backend to train faster.
```shell
!cd Atari-DQN && python src/trainer.py --config-name=your_training_configuration_file_name home_directory=/content/Atari-DQN/
```

#### Testing
Follow the steps that are shown in the Testing section related to installation on servers or local machines. In addition, make sure that the field `in_colab` is set to `false`.
If you use a different configuration file name, change the code in the cell block by changing the `--config-name` argument to the name of your testing configuration file.
```shell
!cd Atari-DQN && python src/tester.py --config-name=your_testing_configuration_file_name home_directory=/content/Atari-DQN/
```

## Configuration file override
If you don't want to change the configuration files directly, you can simply override the values of the configuration file when launching the training/testing commands. To do so, you just need to add as an argument the name of the field of the configuration than you want to change and its value.

For example, if you want to override the `output_model_file` field without changing directly the configuration, you can launch the following command
```shell
python src/trainer.py --config-name=your_training_configuration_file_name output_model_file=your_custom_output_file_name
```