On HPC:

```
sbatch run.sbatch
```

Locally/on frontend-server:
```
ssh -L 9932:localhost:9932 -J service@imageseg.scads.ai,s3811141@login1.alpha.hpc.tu-dresden.de s3811141@i8016
```

Adapt jump-host-path and node.
