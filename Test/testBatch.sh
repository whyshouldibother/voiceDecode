for folders in ./testCommands/*; do
    for file in "$folders"/*; do
        echo "$file";
        ./predict "$file"
    done;
done
