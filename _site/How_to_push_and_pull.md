# Setup connection with GitHub

> [!IMPORTANT]
> Ask Adrian for an access token.

On the HPC cluster, run the following command with your username and the access token as password:
```
git clone https://github.com/bunzela/AIzymes
```

To avoid entering the access token too often, use the following command:
```
git config --global credential.helper 'cache --timeout=31536000' # Sets cache for 1 year
```

# Push and Pull
Use the [push_to_github.sh](https://github.com/bunzela/AIzymes/blob/main/push_to_github.sh) and [pull_from_github.sh](https://github.com/bunzela/AIzymes/blob/main/pull_from_github.sh) scripts to push and pull respectively.

Note: You need to add every newly created file using
```
git add <file>
```

