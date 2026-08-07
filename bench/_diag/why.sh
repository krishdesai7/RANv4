#!/bin/bash
echo "login_shell=$(shopt -q login_shell && echo yes || echo no)"
echo "interactive(\$-)=$-"
echo "BASH_ENV=${BASH_ENV:-<unset>}"
echo "juliaup_in_PATH=$(case ":$PATH:" in *juliaup*) echo YES-bashrc-was-sourced;; *) echo no;; esac)"
echo "SHELL=$SHELL"
