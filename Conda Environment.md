I am using Anaconda to manage the python environments, building off of the commands used in this [article](https://towardsdatascience.com/managing-project-specific-environments-with-conda-406365a539ab)

### Create an environment
```bash
conda env create --prefix ./env --file environment.yml
```

### Activate environment
```bash
conda activate ./env
```

### Update environment
```bash
conda env update --prefix ./env --file environment.yml --prune
```

### Rebuild environment
```bash
conda env create --prefix ./env --file environment.yml --force
```
### Deactivate environment
```bash
conda deactivate
```
