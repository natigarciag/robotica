rsync -r ./{*.py,*.sh,datasets,runAvoid,*.joblib} $3@"192.168.1.$1":"~/$2/"

# echo "192.168.1.$1"
# echo "~/$2/"