set -e 
cd ~

if [ ! -d ~/env ]; then
	mkdir env
fi

cd env
virtualenv -p python3 zzn

source zzn/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt
