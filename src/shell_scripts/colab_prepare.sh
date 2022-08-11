echo "Installing packages for colab..."

# install repo package
cd /content/Atari-Deep-RL
pip install -e .
pip install -r requirements.txt
cd /content/

# install x11-utils
apt-get install x11-utils

# install python opengl
apt-get install -y xvfb python-opengl ffmpeg

## download atari roms from the web and copy them to gym folder
#wget http://www.atarimania.com/roms/Roms.rar
#unrar x -Y "Roms.rar" "roms/"
#python -m atari_py.import_roms "roms/"
#
## remove roms
#rm -r "roms/"
#rm "Roms.rar"

