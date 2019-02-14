#!/bin/bash

user_name='Travis CI'
user_email='travis@travis-ci.org'

if [[ "$user_name" != "$(git log -1 --pretty=format:'%an')" ]]; then
    git checkout --orphan $TRAVIS_BRANCH-pdf
    git rm -rf .
    git add -f main.pdf
    git -c user.name="$user_name" -c user.email="$user_email" commit -m "Add built pdf"
    git push https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf
fi
