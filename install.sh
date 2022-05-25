conda create -n ref_prior python=3.8
conda activate ref_prior
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
