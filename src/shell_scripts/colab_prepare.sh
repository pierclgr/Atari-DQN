# install x11-utils
echo "Installing packages for colab..."
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

# download atari roms from the web and copy them to gym folder
wget http://www.atarimania.com/roms/Roms.rar
unrar x -Y "Roms.rar" "roms/"
python -m atari_py.import_roms "roms/"

rm -r "roms/"
rm "Roms.rar"