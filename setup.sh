pyenv install --skip-existing
export POETRY_VIRTUALENVS_IN_PROJECT=true && poetry install

aws s3 mb s3://rishabhmalviya---cancer-classification --region us-west-1

git init
git add .
git commit -m 'cookiecutter skeleton init'
git branch -M main
git remote add origin git@github.com:RishabhMalviya/cancer-classification.git
git push --set-upstream origin main

