# mount NFS shares

```
MOunt the NFS share on all nodes of the cluster

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
