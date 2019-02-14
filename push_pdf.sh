#!/bin/bash

git add main.pdf
git -c user.name='travis' -c user.email='travis' commit -m "Add built pdf"
git push https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:master
