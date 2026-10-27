for folders in ../data/wakeup/testCommands/*; do
    for file in "$folders"/*; do
        echo "$file";
        ./predict "$file"
    done;
done
