
# before
# $ gcloud compute scp setup.sh nvijayakumar@gpu-instance:~
# $ gcloud compute scp id_ed25519 nvijayakumar@gpu-instance:~
#

sudo apt update
sudo apt install -y nvidia-driver-455

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -h
./miniconda3/bin/conda init

source .bashrc

conda install -y numpy pandas scikit-learn matplotlib jupyterlab
conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

git config --global user.name "nithvijay"
git config --global user.email "nithin2463@example.com"

# $ jupyter lab --no-browser --port=8889