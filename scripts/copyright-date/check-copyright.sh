#!/usr/bin/env bash

# Start of configuration
preamble="Copyright +\([cC]\) +"
postamble=",? +Advanced +Micro +Devices, +Inc\. +All +rights +reserved\."
# End of configuration

print_help() { printf \
"\033[36musuage\033[0m: \033[33mcheck_year.sh [-h] [-u] [-a] [-d <SHA>] [-k] [-v]\033[0m
\033[36mdescription\033[0m: Checks for if the copyright year in the staged files is up to date and displays the files with out-of-date copyright statements. Exits with '0' if successful and with '1' if something is out of date.
\033[36moptions\033[0m:
  \033[34m-h\033[0m       Displays this message.
  \033[34m-u\033[0m       Automatically updates the copyright year
  \033[34m-a\033[0m       Automatically applies applies the changes to current staging environment. Implies '-u'.
  \033[34m-d <SHA>\033[0m Compare using the diff of a hash.
  \033[34m-k\033[0m       Compare using the fork point: where this branch and 'remotes/origin/HEAD' diverge.
  \033[34m-v\033[0m       Verbose output.
Use '\033[33mgit config --local hooks.updateCopyright <true|false>\033[0m' to automatically apply copyright changes on commit.
"
}

# argument parsing
apply=false
update=false
verbose=false
forkdiff=false

while getopts "auhvkqd:" arg; do 
    case $arg in
        a) update=true;apply=true;;
        u) update=true;;
        v) verbose=true;;
        k) forkdiff=true;;
        q) quiet=true;;
        d) diff_hash=${OPTARG};;
        h) print_help; exit;;
        *) print help; exit 1;;
    esac
done
# update & apply copyright when hook config is set
if [ "$(git config --get --type bool --default false hooks.updateCopyright)" = "true" ]; then
    apply=true;update=true;
fi

# If set, check all files changed since the fork point
if $forkdiff; then
    branch="$(git rev-parse --abbrev-ref HEAD)"
    remote="$(git config --local --get "branch.$branch.remote" || echo 'origin')"
    source_commit="remotes/$remote/HEAD"

    # don't use fork-point for finding fork point (lol)
    # see: https://stackoverflow.com/a/53981615
    diff_hash="$(git merge-base "$source_commit" "$branch")"
fi

if [ -n "${diff_hash}" ]; then
    printf "Using diff: %s\n" "${diff_hash}"
else
    diff_hash="HEAD"
fi
diff_origin="$(git diff "${diff_hash}" --cached --name-only)"

# Current year
year="$(date +%Y)"

outdated_copyright=()
uptodate_copyright=()
notfound_copyright=()

printf "Checking if copyright statements are up-to-date... "
for i in $diff_origin; do
    # If copyright exists:
    if grep -q -E "$preamble([0-9]{4}-)?[0-9]{4}$postamble" -i "$i"; then
        #                    .----^---.  .---^--.
        #                    start year  end year
        # If copyright not up to date:
        if ! grep -q -E "$preamble([0-9]{4}-)?$year$postamble" -i "$i"; then
            #                      .----^---.    ^-.
            #                      start year  current year
            outdated_copyright+=("$i")
        else
            uptodate_copyright+=("$i")
        fi
    else
        notfound_copyright+=("$i")
    fi
done

printf "\033[32mDone!\033[0m\n\n"
if $verbose; then
    if (( ${#notfound_copyright[@]} )); then
        printf "\033[36mCouldn't find the copyright in the following files:\033[0m\n"
        for i in "${notfound_copyright[@]}"; do
            echo "- $i"
        done
        printf "\n"
    fi

    if (( ${#uptodate_copyright[@]} )); then
        printf "\033[36mThe copyright statement was already up to date in the following files:\033[0m\n"
        for i in "${uptodate_copyright[@]}"; do
            echo "- $i"
        done
        printf "\n"
    fi
fi

if ! (( ${#outdated_copyright[@]} )); then
    exit 0
fi

printf \
"\033[31m==== COPYRIGHT OUT OF DATE ====\033[0m
\033[36mThe following files need to be updated:\033[0m\n"
for i in "${outdated_copyright[@]}"; do
    echo "- $i"
done

# If we don't need to update, we early exit.
if ! $update; then
    printf \
"\nRun '\033[33mscripts/copyright-date/check-copyright.sh -u\033[0m' to update the copyright statement(s). See '-h' for more info.
Or use '\033[33mgit config --local hooks.updateCopyright true\033[0m' to automatically update copyrights on pre-commit\n"
    exit 1
fi

if $apply; then
    printf "Updating copyrights and staging changes... "
else
    printf "Updating copyrights... "
fi

for i in "${outdated_copyright[@]}"; do
    # Update copyright files...
    sed -E "s/($preamble)([0-9]{4})(-[0-9]{4})?($postamble)/\1\2-$year\4/gi" -i "$i"
    #            .-^-.     .---^----.   .--^---. .-^-.       .-^   ^-.
    #             \1       start year   end year  \4    start year   current year

    # If needed, apply the first hunk to the staging environment.
    # This is a reasonable assumption since the copyright header should be at the top.
    if $apply; then
        git diff -- "$i" | awk 'second && /^@@/ {exit} /^@@/ {second=1} {print}' | git apply --cached
    fi
done

printf "\033[32mDone!\033[0m\n"
