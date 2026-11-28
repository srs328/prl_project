# UMass HPC Usage

```
bsub -q long -R 'rusage[mem=8G]' -W 72:00 "echo '' | /share/bin/code tunnel user login --provider microsoft && while true; do /share/bin/code tunnel; sleep 5; done"
```