# Initialize git repo
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m 'cookiecutter skeleton init'
    git branch -M main
    git remote add origin git@github.com:RishabhMalviya/cancer-classification.git
    git push --set-upstream origin main
fi


# Initialize S3 bucket for MLFlow
if ! aws s3 ls "s3://rishabhmalviya---cancer-classification" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Bucket already exists"
else
    aws s3 mb s3://rishabhmalviya---cancer-classification --region us-west-1
fi

# Initialize python virtual environment
pyenv install --skip-existing
poetry install --all-groups

# Installations for remote SSH development
sudo apt-get install openssh-server
