cd ~

if [ ! -d ~/env ]; then
	mkdir env
fi

cd env
virtualenv -p python3 zzn

pip3 install --upgrade pip
pip3 install numpy pandas scikit-learn==0.20rc1
