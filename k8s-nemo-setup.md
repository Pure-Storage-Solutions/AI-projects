# k8s install - Kubespray

```
git clone -b release-2.24 https://github.com/kubernetes-sigs/kubespray.git
cd kubespray
export KUBE_CONTROL_HOSTS=1
cp -rfp inventory/sample inventory/ragcluster
declare -a IPS=(192.168.125.7 192.168.125.26)
pip3 install ruamel.yaml
pip3 install pYAML
CONFIG_FILE=inventory/ragcluster/hosts.yaml python3 contrib/inventory_builder/inventory.py ${IPS[@]}
pip install -r requirements.txt
apt-get install sshpass

inventory/ragcluster/group_vars/k8s_cluster/k8s-cluster.yml

ansible-playbook -i inventory/ragcluster/hosts.yaml  --become --become-user=root --extra-vars "kubeconfig_localhost=true kubectl_localhost=true" cluster.yml -u ubuntu --ask-pass 
```
# NVIDIA gpu operator

```
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update

helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator
```


# mount NFS shares

```
mount 192.168.125.90:/wikipedia /mnt/wikipedia

```

# local-path-storage

Use Rancher local-storage provisioner

```
cd /mnt/wikipedia/rag-bench/zilliz/k8s/
k apply -f local-path-storage.yaml
```

# set default sc

```
kubectl label --overwrite ns local-path-storage pod-security.kubernetes.io/warn=privileged pod-security.kubernetes.io/enforce=privileged

kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

```

# Milvus install 

```
helm repo add milvus https://zilliztech.github.io/milvus-helm/ && helm repo update

cd /mnt/wikipedia/rag-bench/zilliz/k8s/

helm upgrade --install milvus-deployment milvus/milvus -f s3-milvus.yaml 

update httpNumThreads tp 150 

kubectl edit cm milvus-deployment-pulsar-proxy

```

# NVIDIA-RAG pipeline

https://docs.nvidia.com/ai-enterprise/rag-llm-operator/24.3.0/pipelines.html

```
kubectl create namespace rag-operator
kubectl label --overwrite ns rag-operator pod-security.kubernetes.io/warn=privileged pod-security.kubernetes.io/enforce=privileged

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
   && helm repo update

helm repo add rag-operator https://helm.ngc.nvidia.com/ohlfw0olaadg/ea-participants \
  --username "\$oauthtoken" --password OGlsMjEyY3BpbWNmcGVpNWttOWoyYWt1Mjc6MGM2ZmM0OWMtYzEyYy00Nzk2LTg4ZDAtMWUyNmFiY2I0Njgz


kubectl create secret -n rag-operator docker-registry ngc-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password=OGlsMjEyY3BpbWNmcGVpNWttOWoyYWt1Mjc6MGM2ZmM0OWMtYzEyYy00Nzk2LTg4ZDAtMWUyNmFiY2I0Njgz

helm install rag-operator rag-operator/rag-operator -n rag-operator

kubectl get pods -n rag-operator


kubectl create namespace rag-sample
kubectl label --overwrite ns rag-sample pod-security.kubernetes.io/warn=privileged pod-security.kubernetes.io/enforce=privileged

kubectl create secret -n rag-sample docker-registry ngc-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password=OGlsMjEyY3BpbWNmcGVpNWttOWoyYWt1Mjc6MGM2ZmM0OWMtYzEyYy00Nzk2LTg4ZDAtMWUyNmFiY2I0Njgz
    
ngc registry resource download-version ohlfw0olaadg/ea-participants/rag-sample-pipeline:24.03

cd rag-sample-pipeline_v24.03

kubectl apply -f examples/pvc-embedding.yaml -n rag-sample
kubectl apply -f examples/pvc-inferencing.yaml -n rag-sample
kubectl apply -f examples/pvc-pgvector.yaml -n rag-sample

kubectl get pvc -n rag-sample

kubectl create secret -n rag-sample generic ngc-api-secret \
    --from-literal=NGC_CLI_API_KEY=OGlsMjEyY3BpbWNmcGVpNWttOWoyYWt1Mjc6MGM2ZmM0OWMtYzEyYy00Nzk2LTg4ZDAtMWUyNmFiY2I0Njgz

kubectl create secret -n rag-sample generic hf-secret \
    --from-literal=HF_USER=unni.pure \
    --from-literal=HF_PAT=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

The vLLM works but trtLLM is not still working. 
Update the correct production version in the config/samples/helmpipeline_app_vllm.yaml file

kubectl apply -f config/samples/helmpipeline_app_vllm.yaml  -n rag-sample

```







