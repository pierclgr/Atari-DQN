# install x11-utils
apt-get install x11-utils

# install pyglet
pip install pyglet

# install python opengl
apt-get install -y xvfb python-opengl ffmpeg

# install pyvirtualdisplay
pip install pyvirtualdisplay

# install colabgymrender to render the agent on colab
pip install -U colabgymrender

# install gym environments
pip install gym[box2d]
pip install gym[atari]

# donwload atary roms from the web and copy them to gym folder
wget http://www.atarimania.com/roms/Roms.rar
unrar x -Y "/content/Roms.rar" "/content/roms/"
python -m atari_py.import_roms "/content/roms/"