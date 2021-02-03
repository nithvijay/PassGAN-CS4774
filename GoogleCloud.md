# Google Cloud Platform Cheatsheet

## Setup 
1. Make Google Cloud Platform account with credit card attached
2. Install GCP SDK
3. Create vm instance
   1. Install programs
   2. Create env and install python packages
4. Set up ssh/tunneling

https://cloud.google.com/storage/docs/reference/libraries#cloud-console

https://cloud.google.com/compute/docs/instances/create-start-preemptible-instance?authuser=1#handle_preemption

https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html




## Projects
```console
$ gcloud projects list
```

## Config
```console
$ gcloud init
$ gcloud config list
$ gcloud config set 
$ gcloud config unset

$ gcloud config set compute/zone us-central1-a
$ gcloud config set compute/region us-central1
```

## Cloud Storage
```console
$ gsutil ls
$ gsutil cp
```

## Compute Engine
```console
$ gcloud compute instances list
$ gcloud compute instances start main-instance
$ gcloud compute instances stop main-instance
$ gcloud compute ssh main-instance #takes a second after starting

$ gcloud compute ssh main-instance -- -L localhost:8888:localhost:8889 #ssh tunneling
$ gcloud compute config-ssh


$ ssh preempt-instance.us-central1-a.python-compute-engine-285704
$ ssh main-instance.us-east1-b.python-compute-engine-285704
```

In Compute Instance
```console
$ export PATH="$HOME/.local/bin:$PATH" #once
$ source Python\ Project/ML_env/bin/activate && jupyter lab --no-browser --port=8889
```

## Github SSH
```console
$ ls -al ~/.ssh #see existing keys
$ ssh-keygen -t ed25519 -C "nithin2463@gmail.com" #make public/private key pair
$ cat ~/.ssh/id_ed25519.pub #copy public key to github
$ eval "$(ssh-agent -s)" #start ssh-agent
$ ssh-add ~/.ssh/id_ed25519 #add private key to the ssh-agent

$ gcloud compute scp id_ed25519 nvijayakumar@gpu-instance:~
$ mv id_ed25519 ~/.ssh
```


## Ubuntu
```console
$ sudo apt update
$ sudo apt install -y git python3-pip python3-venv
$ echo "LS_COLORS=$LS_COLORS:'di=0;36:' ; export LS_COLORS" >> ~/.bashrc

$ sudo shutdown now
```

## Python
```console
$ python3 -m venv ML_env
$ source ML_env/bin/activate
```

## Screen
```console
$ screen
$ screen -r
```

`CTRL + a + d` to detach





