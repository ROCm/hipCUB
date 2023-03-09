#!/usr/bin/env bash

# Start of configuration
preamble="Copyright +\([cC]\) +"
postamble=",? +Advanced +Micro +Devices, +Inc\. +All +rights +reserved\."
# End of configuration

print_help() { printf -- \
"\033[36musuage\033[0m: \033[33mcheck_year.sh [-h] [-u] [-a] [-d <SHA>] [-k] [-v]\033[0m
\033[36mdescription\033[0m: Checks for if the copyright year in the staged files is up to date and displays the files with out-of-date copyright statements. Exits with '0' if successful and with '1' if something is out of date.
\033[36moptions\033[0m:
  \033[34m-h\033[0m       Displays this message.
  \033[34m-u\033[0m       Automatically updates the copyright year
  \033[34m-a\033[0m       Automatically applies applies the changes to current staging environment. Implies '-u' and '-c'.
  \033[34m-a\033[0m       Compare files to the index instead of the working tree.
  \033[34m-d <SHA>\033[0m Compare using the diff of a hash.
  \033[34m-k\033[0m       Compare using the fork point: where this branch and 'remotes/origin/HEAD' diverge.
  \033[34m-q\033[0m       Suppress updates about progress.
  \033[34m-v\033[0m       Verbose output.
Use '\033[33mgit config --local hooks.updateCopyright <true|false>\033[0m' to automatically apply copyright changes on commit.
"
}

# argument parsing
apply=false
update=false
verbose=false
forkdiff=false
quiet=false
cached=false

while getopts "auhvkqcd:" arg; do 
    case $arg in
        a) update=true;apply=true;cached=true;;
        u) update=true;;
        v) verbose=true;;
        k) forkdiff=true;;
        q) quiet=true;;
        c) cached=true;;
        d) diff_hash=${OPTARG};;
        h) print_help; exit;;
        *) print help; exit 1;;
    esac
done

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
    $verbose && printf -- "Using base commit: %s\n" "${diff_hash}"
else
    diff_hash="HEAD"
fi

# Current year
year="$(date +%Y)"

diff_opts=(-z --name-only)
git_grep_opts=(-z --extended-regexp --ignore-case --no-recursive -I)
if $cached; then
    diff_opts+=(--cached)
    git_grep_opts+=(--cached)
fi

! $quiet && printf -- "Checking if copyright statements are up-to-date... "
mapfile -d $'\0' changed_files < <(git diff-index "${diff_opts[@]}" "$diff_hash" | LANG=C.UTF-8 sort -z)

if (( ${#changed_files[@]} )); then
    mapfile -d $'\0' found_copyright < <(                                                                \
        git grep "${git_grep_opts[@]}" --files-with-matches -e "$preamble([0-9]{4}-)?[0-9]{4}$postamble" \
            -- "${changed_files[@]}" |                                                                   \
        LANG=C.UTF-8 sort -z)
else
    found_copyright=()
fi

if (( ${#found_copyright[@]} )); then
    mapfile -d $'\0' outdated_copyright < <(                                                           \
        git grep "${git_grep_opts[@]}" --files-without-match -e "$preamble([0-9]{4}-)?$year$postamble" \
            -- "${found_copyright[@]}" |                                                               \
        LANG=C.UTF-8 sort -z)
else
    outdated_copyright=()
fi

! $quiet && printf -- "\033[32mDone!\033[0m\n"
if $verbose; then
    # Compute the files that don't have a copyright as the set difference of
    # `changed_files - `found_copyright`
    mapfile -d $'\0' notfound_copyright < <(                                   \
        printf -- '%s\0' "${changed_files[@]}" |                               \
        LANG=C.UTF-8 comm -z -23 - <(printf -- '%s\0' "${found_copyright[@]}"))

    if (( ${#notfound_copyright[@]} )); then
        printf -- "\033[36mCouldn't find a copyright statement in %d file(s):\033[0m\n" \
            "${#notfound_copyright[@]}"
        printf -- '  - %q\n' "${notfound_copyright[@]}"
    fi

    # Similarly the up-to-date files are the difference of `found_copyright` and `outdated_copyright`
    mapfile -d $'\0' uptodate_copyright < <(                                       \
        printf -- '%s\0' "${found_copyright[@]}" |                                 \
        LANG=C.UTF-8 comm -z -23 - <(printf -- '%s\0' "${outdated_copyright[@]}"))

    if (( ${#uptodate_copyright[@]} )); then
        printf -- "\033[36mThe copyright statement was already up to date in %d file(s):\033[0m\n" \
            "${#uptodate_copyright[@]}"
        printf -- '  - %q\n' "${uptodate_copyright[@]}"
    fi
fi

if ! (( ${#outdated_copyright[@]} )); then
    exit 0
fi

printf -- \
"\033[31m==== COPYRIGHT OUT OF DATE ====\033[0m
\033[36m%d file(s) need(s) to be updated:\033[0m\n" "${#outdated_copyright[@]}"
printf -- '  - %q\n' "${outdated_copyright[@]}"

# If we don't need to update, we early exit.
if ! $update; then
    printf -- \
"\nRun '\033[33mscripts/copyright-date/check-copyright.sh -u\033[0m' to update the copyright statement(s). See '-h' for more info,
or set '\033[33mgit config --local hooks.updateCopyright true\033[0m' to automatically update copyrights when committing.\n"
    exit 1
fi

if $apply; then
    ! $quiet && printf -- "Updating copyrights and staging changes... "
else
    ! $quiet && printf -- "Updating copyrights... "
fi

sed=(sed --regexp-extended --separate "s/($preamble)([0-9]{4})(-[0-9]{4})?($postamble)/\1\2-$year\4/gi")
# Just update the files in place if only touching the working-tree
if ! $apply; then
    "${sed[@]}" -i "${outdated_copyright[@]}"
    printf -- "\033[32mDone!\033[0m\n"
    exit 0
fi

# Make a tree out of the current contents of the index, as the base for the updates
if ! old_tree="$(git write-tree)"; then
    printf -- "\033[31mFailed to write out current index, is merge in progress?\033[0m"
    exit 1
fi

# Make a separate index so we're not messing up what's already cached, creating a new tree is more
# convenient through an index.
temp_index="$(git rev-parse --git-dir)/copyright-check-index"
if ! git read-tree "--index-output=$temp_index" "$old_tree"; then
    exit 1
fi

# Cleanup temp_index when the script exits
finish () {
    rm -f "$temp_index"
}
# The trap will be invoked whenever the script exits, even due to a signal, this is a bash only
# feature
trap finish EXIT

# Update the outdated files in the temporary index
for i in "${outdated_copyright[@]}"; do
    # Read in the current mode of the file (permissions, etc)
    IFS=' ' read -r -d $'\0' mode < <(git ls-files -z --stage --cached -- "$i" |
        cut -z --delimiter=' ' --fields=1)

    # Run sed on it to fix the copyright and save to a blob
    blob="$(git cat-file blob ":$i" | "${sed[@]}" | git hash-object -w --path "$i" --stdin)"

    # Output the blob, its mode and its path, to add it to the index
    printf -- '%s blob %s\t%s\0' "$mode" "$blob" "$i"
done | GIT_INDEX_FILE="$temp_index" git update-index -z --index-info

# Write out the temporary to a tree, so that patches can be generated from it
if ! new_tree="$(GIT_INDEX_FILE="$temp_index" git write-tree)"; then
    exit 1;
fi

if ! git diff-tree -U0 "$old_tree" "$new_tree" | git apply --unidiff-zero; then
    printf -- "\033[31mFailed to apply changes to working tree.
Perhaps the fix is already applied, but not yet staged?\n\033[0m"
    exit 1
fi

if ! git diff-tree -U0 "$old_tree" "$new_tree" | git apply --unidiff-zero --cached; then
    printf -- "\033[31mFailed to apply change to the index.\n\033[0m"
    exit 1
fi

! $quiet && printf -- "\033[32mDone!\033[0m\n"
exit 0
