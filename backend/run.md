uvicorn test:app --host 0.0.0.0 --port 9932
ssh -L 9932:localhost:9932 -J service@imageseg.scads.ai,s3811141@login1.alpha.hpc.tu-dresden.de s3811141@i8016
