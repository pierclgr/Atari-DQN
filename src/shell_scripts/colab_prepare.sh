echo "Installing packages for colab..."

# install repo package
cd /content/Atari-Deep-RL
pip install -e .
pip install -r requirements.txt
pip install gym[atari]
cd /content/

# install x11-utils
apt-get install x11-utils

# install python opengl
apt-get install -y xvfb python-opengl ffmpeg