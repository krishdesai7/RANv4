# Make `module` usable inside a batch script. Sourced, never executed.
#
# `module` is a shell function rather than a program, and a SLURM batch script
# does not get it: the first `module load ...` dies with
# `command not found: module`, which under `set -e` takes the whole job with it
# after the expensive part has already run.
#
# A batch script runs zsh *non-interactive and non-login*, which
# reads only `/etc/zshenv` and `~/.zshenv`. The rc files -- where a site defines
# `module` -- are interactive-only, and the profile files are login-only. So
# `whence module` at a prompt finds the function and the same check inside the
# job does not. On a login node:
#
#     whence module          # the Lmod function
#     zsh -c 'whence module' # nothing: this is what the batch script sees
#
# Observed as `slurm_script:65: command not found: module`.
#
# `MODULESHOME` is an ordinary exported *variable*, so unlike the function it
# does survive into the job; it is the thing to trust, with Perlmutter's path as
# the fallback for the case where even that is missing.
#
# `whence` rather than `typeset -f`: it finds a command, builtin or alias too,
# so a site that ships `module` as something other than a function is left
# alone instead of being re-sourced over.

if ! whence module > /dev/null 2>&1; then
  # `MODULESHOME` first because it is what the site actually set. The rest are
  # fallbacks for the case where even that did not survive: the generic Lmod
  # prefix, then Cray's, which is where Perlmutter keeps it.
  lmod_candidates=(
    ${MODULESHOME:+"${MODULESHOME}/init/zsh"}
    /usr/share/lmod/lmod/init/zsh
    /opt/cray/pe/lmod/lmod/init/zsh
  )

  for lmod_init in "${lmod_candidates[@]}"; do
    if [[ -r "${lmod_init}" ]]; then
      source "${lmod_init}"
      break
    fi
  done

  if ! whence module > /dev/null 2>&1; then
    print -u2 "Cannot initialise the module system: \`module\` is undefined and"
    print -u2 "none of these were readable:"
    print -u2 ${(F)lmod_candidates}
    print -u2 "Check \`echo \$MODULESHOME\` on a login node and export it."
    exit 1
  fi
fi
