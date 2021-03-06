#!/bin/bash

user_name="$GH_USER_NAME"
user_email="$GH_USER_EMAIL"

if [[ "$user_name" != "$(git log -1 --pretty=format:'%an')" ]]; then
    git checkout --orphan $TRAVIS_BRANCH-pdf
    git rm -rf .
    git checkout master main.pdf
    git add -f main.pdf
    git -c user.name="$user_name" -c user.email="$user_email" commit -m "Add built pdf"
    git push -f git@github.com:$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf --quiet
fi
