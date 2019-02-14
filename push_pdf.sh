#!/bin/bash

user_name='travis'
user_email='oyvindschoyen@gmail.com'

if [[ "$user_name" != "$(git log -1 --pretty=format:'%an')" ]]; then
    git add main.pdf
    git -c user.name=$user_name -c user.email=$user_email commit -m "Add built pdf"
    git push https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:master
fi
